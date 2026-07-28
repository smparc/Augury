# TimescaleDB access for the analytics layer.
#
# Read-only except for `write_granger_results`. The analytics module consumes
# what the pipeline produced and writes back only its own conclusions.

suppressPackageStartupMessages({
  library(DBI)
  library(RPostgres)
  library(dplyr)
})

#' Connection parameters from the repo-root .env, matching the other services.
augury_connection_params <- function(env_file = find_env_file()) {
  defaults <- list(
    host = "localhost", port = 5432L,
    dbname = "augury", user = "augury", password = "augury_local_dev"
  )

  if (is.null(env_file) || !file.exists(env_file)) {
    return(defaults)
  }

  lines <- readLines(env_file, warn = FALSE)
  lines <- lines[!grepl("^\\s*(#|$)", lines)]

  values <- list()
  for (line in lines) {
    parts <- strsplit(line, "=", fixed = TRUE)[[1]]
    if (length(parts) >= 2) {
      key <- trimws(parts[1])
      value <- trimws(paste(parts[-1], collapse = "="))
      values[[key]] <- value
    }
  }

  list(
    host = values[["POSTGRES_HOST"]] %||% defaults$host,
    port = as.integer(values[["POSTGRES_PORT"]] %||% defaults$port),
    dbname = values[["POSTGRES_DB"]] %||% defaults$dbname,
    user = values[["POSTGRES_USER"]] %||% defaults$user,
    password = values[["POSTGRES_PASSWORD"]] %||% defaults$password
  )
}

find_env_file <- function() {
  current <- normalizePath(getwd(), mustWork = FALSE)
  for (depth in seq_len(8)) {
    candidate <- file.path(current, ".env")
    if (file.exists(candidate)) return(candidate)
    parent <- dirname(current)
    if (identical(parent, current)) break
    current <- parent
  }
  NULL
}

#' Open a connection. Caller is responsible for `DBI::dbDisconnect`.
augury_connect <- function(params = augury_connection_params()) {
  DBI::dbConnect(
    RPostgres::Postgres(),
    host = params$host, port = params$port, dbname = params$dbname,
    user = params$user, password = params$password
  )
}

#' TRUE when a connection can be opened. Used by the report to degrade
#' gracefully into a "database unavailable" section rather than erroring out
#' halfway through rendering.
augury_available <- function() {
  tryCatch({
    connection <- augury_connect()
    on.exit(DBI::dbDisconnect(connection))
    TRUE
  }, error = function(e) FALSE)
}

fetch_markets <- function(connection, tracked_only = TRUE) {
  sql <- "SELECT * FROM markets"
  if (tracked_only) sql <- paste(sql, "WHERE tracked")
  DBI::dbGetQuery(connection, paste(sql, "ORDER BY market_id"))
}

fetch_signals <- function(connection, market_id, model_version = NULL) {
  if (is.null(model_version)) {
    DBI::dbGetQuery(
      connection,
      "SELECT ts, s_t, p_hat, calibration, n_posts, half_life_seconds
         FROM signals WHERE market_id = $1 ORDER BY ts",
      params = list(market_id)
    )
  } else {
    DBI::dbGetQuery(
      connection,
      "SELECT ts, s_t, p_hat, calibration, n_posts, half_life_seconds
         FROM signals WHERE market_id = $1 AND model_version = $2 ORDER BY ts",
      params = list(market_id, model_version)
    )
  }
}

#' Prices, with a usable mid computed at the database.
#'
#' COALESCE order matters: the quote midpoint is preferred over the last trade,
#' because a last-trade price on a thin market can be hours stale while the
#' quote has moved.
fetch_prices <- function(connection, market_id) {
  DBI::dbGetQuery(
    connection,
    "SELECT ts, source, yes_price, yes_bid, yes_ask, spread, depth_bid, depth_ask, volume,
            COALESCE((yes_bid + yes_ask) / 2.0, yes_price) AS mid
       FROM market_prices WHERE market_id = $1 ORDER BY ts",
    params = list(market_id)
  )
}

fetch_sim_prices <- function(connection, market_id, run_id = "live") {
  DBI::dbGetQuery(
    connection,
    "SELECT ts, sim_price, b, q_yes, q_no FROM sim_prices
      WHERE market_id = $1 AND run_id = $2 ORDER BY ts",
    params = list(market_id, run_id)
  )
}

fetch_scores <- function(connection) {
  DBI::dbGetQuery(connection, "SELECT * FROM scores ORDER BY computed_at DESC")
}

#' Write FDR-corrected Granger results back for the API to serve.
write_granger_results <- function(connection, results_frame, run_id, model_version) {
  if (nrow(results_frame) == 0) {
    return(invisible(0L))
  }

  rows <- results_frame %>%
    mutate(run_id = run_id, model_version = model_version)

  for (i in seq_len(nrow(rows))) {
    row <- rows[i, ]
    DBI::dbExecute(
      connection,
      "INSERT INTO granger_results (run_id, market_id, model_version, direction, lag_order,
                                    lag_criterion, n_obs, f_statistic, p_value, p_value_adj,
                                    significant, adf_p_signal, adf_p_price, liquidity_control)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
       ON CONFLICT (run_id, market_id, model_version, direction) DO UPDATE SET
           p_value = EXCLUDED.p_value,
           p_value_adj = EXCLUDED.p_value_adj,
           significant = EXCLUDED.significant",
      params = list(
        row$run_id, row$market_id, row$model_version, row$direction, row$lag_order,
        row$lag_criterion, row$n_obs, row$f_statistic, row$p_value, row$p_value_adj,
        row$significant, row$adf_p_signal, row$adf_p_price, row$liquidity_control
      )
    )
  }

  invisible(nrow(rows))
}

`%||%` <- function(x, y) if (is.null(x)) y else x
