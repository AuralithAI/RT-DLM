# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

### 1. **Do NOT open a public issue**
Security vulnerabilities should be reported privately to prevent exploitation.

### 2. **Contact us directly**
- Email: contact@auralithai.com
- Or use GitHub's private vulnerability reporting feature

### 3. **Include the following information**
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### 4. **Response timeline**
- **Initial response**: Within 48 hours
- **Status update**: Within 7 days
- **Fix timeline**: Depends on severity
  - Critical: 24-48 hours
  - High: 7 days
  - Medium: 30 days
  - Low: 90 days

## Security Measures

This repository implements the following security measures:

### Code Protection
- ✅ Branch protection on `main`
- ✅ Required pull request reviews
- ✅ CODEOWNERS file for mandatory reviews
- ✅ Signed commits recommended

### Automated Security
- ✅ CodeQL analysis on every PR
- ✅ Dependabot for dependency updates
- ✅ Secret scanning with TruffleHog & Gitleaks
- ✅ Dependency vulnerability review

### Best Practices
- No secrets in code (use environment variables)
- Minimal permissions in workflows
- Regular dependency updates
- Security-focused code review

## Signed Commits

We recommend using signed commits for all contributions:

```bash
# Configure GPG signing
git config --global commit.gpgsign true
git config --global user.signingkey YOUR_GPG_KEY_ID
```

## Acknowledgments

We appreciate responsible disclosure and will acknowledge security researchers who help improve our security.
