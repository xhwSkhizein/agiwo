# Public Site Deployment

## GitHub Pages

1. Open the repository on GitHub.
2. Click **Settings**.
3. In the left sidebar, open **Pages** under **Code and automation**.
4. In **Build and deployment**, set **Source** to **GitHub Actions**.
5. This repository deploys Pages only from `main`. The `Public Docs` workflow will build on the feature branch, but the deploy job runs only after the PR is merged into `main`.
6. After merge, open the **Actions** tab and confirm the `Public Docs` workflow finishes successfully.

## Custom Domain

Set the custom domain to:

`docs.agiwo.o-ai.tech`

The static artifact already includes a `CNAME` file with the same domain.

## Cloudflare DNS

Create a `CNAME` record:

- If the Cloudflare zone is `o-ai.tech`, set **Name** to `docs.agiwo`
- Or enter the full hostname `docs.agiwo.o-ai.tech` if you prefer full names in the UI
- Target: `xhwSkhizein.github.io`

Keep the record DNS-only for the initial verification pass.

## HTTPS

After GitHub Pages accepts the custom domain and issues the certificate:

1. Return to **Settings** -> **Pages**
2. Confirm the domain is still `docs.agiwo.o-ai.tech`
3. Enable **Enforce HTTPS**

## Post-deploy checks

- Open `https://docs.agiwo.o-ai.tech/`
- Open `https://docs.agiwo.o-ai.tech/robots.txt`
- Open `https://docs.agiwo.o-ai.tech/sitemap-index.xml`
- Verify the domain in Google Search Console
- Submit the sitemap in Search Console
- Update the GitHub repository Website field to `https://docs.agiwo.o-ai.tech`
