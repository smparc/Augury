#!/bin/sh
# Apply docker/migrations/*.sql in filename order, exactly once each.
#
# Each file runs inside a single transaction with ON_ERROR_STOP, so a migration
# that fails halfway leaves the database untouched rather than half-migrated.
# Applied versions are recorded in schema_migrations (created by 001).

set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENV_FILE="$ROOT/.env"

if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
fi

DB_USER="${POSTGRES_USER:-augury}"
DB_NAME="${POSTGRES_DB:-augury}"
SERVICE=timescaledb
COMPOSE="docker compose --env-file $ENV_FILE -f $ROOT/docker/docker-compose.yml"

psql_do() {
    $COMPOSE exec -T "$SERVICE" psql -v ON_ERROR_STOP=1 -U "$DB_USER" -d "$DB_NAME" "$@"
}

echo "==> waiting for $SERVICE to accept connections"
i=0
until $COMPOSE exec -T "$SERVICE" pg_isready -U "$DB_USER" -d "$DB_NAME" >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -gt 60 ]; then
        echo "!! database did not become ready in 60s — is \`make up\` running?" >&2
        exit 1
    fi
    sleep 1
done

is_applied() {
    # Missing schema_migrations (fresh database) means "not applied".
    $COMPOSE exec -T "$SERVICE" psql -tAq -U "$DB_USER" -d "$DB_NAME" \
        -c "SELECT 1 FROM schema_migrations WHERE version = '$1'" 2>/dev/null \
        | grep -q '^1$'
}

applied_count=0
for file in "$ROOT"/docker/migrations/*.sql; do
    version=$(basename "$file" .sql)

    if is_applied "$version"; then
        echo "    skip  $version (already applied)"
        continue
    fi

    echo "==> apply $version"
    psql_do --single-transaction -f - < "$file"
    psql_do -q -c "INSERT INTO schema_migrations (version) VALUES ('$version')
                   ON CONFLICT (version) DO NOTHING"
    applied_count=$((applied_count + 1))
done

echo "==> done — $applied_count migration(s) applied"
