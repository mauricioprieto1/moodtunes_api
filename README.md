# moodtunes_api

Project overview (what it does).

Quickstart:

docker compose -f docker/docker-compose.dev.yml up -d

dotnet run --project src/MoodTunes.Api

Config:

appsettings.Development.json schema (Azure endpoint/key, SQL, Redis).

Use dotnet user-secrets locally; Key Vault in prod.

ML workflow:

python ml/build_mood_dataset.py

Upload to Blob, import into Language Studio, train, deploy, set Azure:Language:* in config.

CI/CD badge + deploy steps.