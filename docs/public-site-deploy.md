# Public Site Deployment

## GitHub Pages

1. Open the repository settings.
2. Open **Pages**.
3. Set **Source** to **GitHub Actions**.
4. After the `Public Docs` workflow succeeds on `main`, confirm the site artifact deploys successfully.

## Custom Domain

Set the custom domain to:

`docs.agiwo.o-ai.tech`

The static artifact already includes a `CNAME` file with the same domain.

## Cloudflare DNS

Create a `CNAME` record:

- Name: `docs.agiwo.o-ai.tech`
- Target: `xhwSkhizein.github.io`

Keep the record DNS-only for the initial verification pass.

## Post-deploy checks

- Open `https://docs.agiwo.o-ai.tech/`
- Open `https://docs.agiwo.o-ai.tech/robots.txt`
- Open `https://docs.agiwo.o-ai.tech/sitemap-index.xml`
- Verify the domain in Google Search Console
- Submit the sitemap in Search Console
- Update the GitHub repository Website field to `https://docs.agiwo.o-ai.tech`
