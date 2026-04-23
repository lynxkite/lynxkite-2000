#!/bin/bash -xeu
# Sets up a Keycloak instance in a Docker container for development purposes.

cd "$(dirname "$0")"

container_name=${KEYCLOAK_CONTAINER_NAME:-keycloak-dev}
image=${KEYCLOAK_IMAGE:-quay.io/keycloak/keycloak:latest}
keycloak_url=${KEYCLOAK_URL:-http://localhost:8080}
admin_user=${KEYCLOAK_ADMIN_USER:-admin}
admin_password=${KEYCLOAK_ADMIN_PASSWORD:-admin}
realm=${KEYCLOAK_REALM:-lynxkite-dev}
client_id=${KEYCLOAK_CLIENT_ID:-lynxkite-web}
frontend_origin=${LYNXKITE_FRONTEND_ORIGIN:-http://localhost:5173}
redirect_uri=${KEYCLOAK_REDIRECT_URI:-$frontend_origin/auth/callback}
dev_username=${KEYCLOAK_DEV_USERNAME:-demo}
dev_password=${KEYCLOAK_DEV_PASSWORD:-demo}

run_kcadm() {
  docker exec "$container_name" /opt/keycloak/bin/kcadm.sh "$@"
}

run_kcadm_with_file() {
  local container_json
  container_json="/tmp/lynxkite-client.json"
  docker cp "$client_json" "$container_name:$container_json"
  run_kcadm "$@" -f "$container_json"
  docker exec "$container_name" rm -f "$container_json" >/dev/null 2>&1 || true
}

client_uuid() {
  run_kcadm get clients -r "$realm" -q clientId="$client_id" --fields id --format csv --noquotes | tail -n 1
}

user_uuid() {
  run_kcadm get users -r "$realm" -q username="$dev_username" --fields id --format csv --noquotes | tail -n 1
}

client_json=$(mktemp)
cat >"$client_json" <<EOF
{
  "clientId": "$client_id",
  "name": "LynxKite Dev Web",
  "enabled": true,
  "protocol": "openid-connect",
  "publicClient": true,
  "standardFlowEnabled": true,
  "directAccessGrantsEnabled": false,
  "serviceAccountsEnabled": false,
  "rootUrl": "$frontend_origin",
  "baseUrl": "$frontend_origin",
  "redirectUris": ["$redirect_uri"],
  "webOrigins": ["$frontend_origin"]
}
EOF
trap 'rm -f "$client_json"' EXIT

docker run --name "$container_name" \
  -d \
  -p 8080:8080 \
  -e KEYCLOAK_ADMIN="$admin_user" \
  -e KEYCLOAK_ADMIN_PASSWORD="$admin_password" \
  -v keycloak_data:$PWD/keycloak_data \
  "$image" \
  start-dev || docker start "$container_name"

until curl -fsS "$keycloak_url/realms/master/.well-known/openid-configuration" >/dev/null; do
  sleep 1
done

run_kcadm config credentials \
  --server "$keycloak_url" \
  --realm master \
  --user "$admin_user" \
  --password "$admin_password"

if ! run_kcadm get "realms/$realm" >/dev/null 2>&1; then
  run_kcadm create realms -s realm="$realm" -s enabled=true
fi

if [[ -z "$(client_uuid)" ]]; then
  run_kcadm_with_file create clients -r "$realm"
fi

run_kcadm_with_file update "clients/$(client_uuid)" -r "$realm"

if [[ -z "$(user_uuid)" ]]; then
  run_kcadm create users -r "$realm" \
    -s username="$dev_username" \
    -s enabled=true \
    -s emailVerified=true
fi

run_kcadm update "users/$(user_uuid)" -r "$realm" \
  -s username="$dev_username" \
  -s enabled=true \
  -s emailVerified=true \
  -s firstName=Demo \
  -s lastName=User \
  -s email="$dev_username@example.local" \
  -s 'requiredActions=[]'

run_kcadm set-password -r "$realm" --username "$dev_username" --new-password "$dev_password"

cat <<EOF

Keycloak dev setup is ready.

Realm: $realm
Client ID: $client_id
Frontend origin: $frontend_origin
Redirect URI: $redirect_uri
Dev user: $dev_username
Dev password: $dev_password

Set these backend env vars before starting LynxKite:
  export LYNXKITE_AUTH_ISSUER=$keycloak_url/realms/$realm
  export LYNXKITE_AUTH_AUDIENCE=$client_id

Keycloak admin console:
  $keycloak_url/admin/

EOF
