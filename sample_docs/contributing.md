# Contributing

## Local setup
Install Go 1.23, Docker and `make`. Run `make dev` to start Postgres, Kafka and MinIO in
Docker Compose, then `make run` to start the gateway on `localhost:8080`.

## Code style
- Run `make lint` (golangci-lint) before pushing.
- Prefer small, focused pull requests; one logical change per PR.
- Public functions need doc comments.

## Tests
`make test` runs unit tests. Integration tests need the Docker Compose stack and run with
`make test-integration`. New endpoints must come with an integration test.

## Commit messages
Use Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
Reference the issue number in the footer, e.g. `Refs: #123`.

## Reviews
Every PR needs one approval from a code owner listed in `CODEOWNERS`. Squash-merge is the
default; keep the PR title in Conventional Commit format because it becomes the commit message.
