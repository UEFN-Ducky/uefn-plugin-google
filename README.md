# Google

Google Gemini API key and Gemini CLI coding agent. After Install, Google appears under Settings → LLMs → Providers & Keys, and Gemini CLI under Coding Agents.

Desktop plugin for [UEFN-Ducky](https://github.com/UEFN-Ducky/UEFN-Ducky) (`google`).
Install or update from **Settings → Store** in the app — do not install from a zip by hand.

## Build

```bash
py scripts/build_zip.py
```

Writes `deploy/google-<version>.ducky-plugin.zip` (scripts/, deploy/ and tests are not packed).

## Test

```bash
py -m pytest
```

`backend/conftest.py` stubs the host's `backend.agent.*` packages so the provider and
CLI adapter can be exercised without the app.

## Secrets

Never commit tokens or keys. The app stores `gemini` locally (DPAPI), not in this package.

## License

MIT. Copyright (c) 2026 Mindful Path Company, LLC. See [LICENSE](LICENSE).
