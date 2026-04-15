# Reco

Flutter-based on-device recycling segmentation.

## Tech
- Flutter + Dart
- ONNX Runtime (Android), custom post-processing in `lib/main_page.dart`

## Security-first Git setup
- Sensitive artifacts are excluded via `.gitignore`:
  - Builds/binaries (`*.apk`, `*.aab`, `/build`, iOS/Android derived)
  - Credentials/env (`.env*`, `*.keystore`, `*.jks`, certificates/keys)
  - Large data/models (`assets/model/**`, `train/**`, `test/**`)

### Quick start
```bash
flutter pub get
flutter build apk --release
```

### Recommended git workflow (HTTPS, no token sharing here)
1) Initialize locally
```bash
git init
git add .
git commit -m "chore: initial secure commit"
git branch -M main
```

2) Create an empty repo on GitHub (Web UI). Do NOT commit tokens/keys.

3) Connect remote and push from your machine
```bash
git remote add origin https://github.com/<YOUR_USERNAME>/<REPO>.git
git push -u origin main
```

### Using a token securely
- Never paste tokens in code or commit history.
- Use your local Git credential manager or environment variables.
- For one-off push in this repo:
  - When `git push` prompts for credentials, enter your GitHub username and
    a Personal Access Token (classic) with minimal scopes (repo).
  - Or use GitHub CLI:
```bash
gh auth login
gh repo create <REPO> --public --source=. --remote=origin --push
```

## Notable code
- `lib/main_page.dart`: model I/O, softmax, class weighting, morphological refinements, can suppression.

## Build outputs
- Android release APK is produced at `build/app/outputs/flutter-apk/app-release.apk` (ignored by git).
