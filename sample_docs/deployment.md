# Deployment Guide

## Environments
| Environment | Cluster        | Deploys            |
|-------------|----------------|--------------------|
| dev         | `k8s-dev-1`    | every merge to main |
| staging     | `k8s-stg-1`    | nightly             |
| production  | `k8s-prod-eu`, `k8s-prod-us` | manual, via release tag |

## Releasing to production
1. Create a release tag `vX.Y.Z` on main.
2. CI builds the container images and pushes them to the registry.
3. Run `make deploy ENV=prod VERSION=vX.Y.Z`. This uses Helm with the `values-prod.yaml` file.
4. The rollout is canary: 5% of traffic for 10 minutes, then 50%, then 100%.
5. If the error rate exceeds 1% during the canary, Argo Rollouts aborts and rolls back automatically.

## Rolling back manually
Run `helm rollback nimbus <REVISION> -n nimbus`. Find the revision with `helm history nimbus -n nimbus`.
Database migrations are forward-only, so a rollback must stay compatible with the newest schema.

## Configuration
All configuration is read from environment variables. Secrets come from Vault through the
`vault-agent` injector. Important variables:

- `NIMBUS_DB_URL` – Postgres connection string
- `NIMBUS_STORAGE_BUCKET` – object storage bucket name
- `NIMBUS_MAX_UPLOAD_MB` – per-file upload limit (default 5120)
- `NIMBUS_KAFKA_BROKERS` – comma-separated broker list

## Database migrations
Migrations live in `db/migrations` and run with `make migrate ENV=<env>` before the deploy.
Each migration must be backwards compatible with the previously deployed version.
