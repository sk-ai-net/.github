# 📦 Repository Move Notice

## sk-ai-net → **SKaiNET-developers**

The **sk-ai-net** GitHub organization has been **moved to `SKaiNET-developers`** to better reflect and serve its target audience: developers, contributors, and integrators working with SKaiNET projects.

✅ **Nothing is lost**  
All repositories, files, commit history, issues, and tags remain intact.

---

## 🔗 What You Need to Do

If you are using this repository locally, **please update your Git remote URL** to point to the new organization.

### 1️⃣ Check your current remote
```bash
git remote -v
```

### 2️⃣ Update the remote URL
Replace `sk-ai-net` with `SKaiNET-developers`:

```bash
git remote set-url origin https://github.com/SKaiNET-developers/<repository-name>.git
```

For SSH users:
```bash
git remote set-url origin git@github.com:SKaiNET-developers/<repository-name>.git
```

### 3️⃣ Verify the change
```bash
git remote -v
```

---

## 🌐 Update Your Links

Please update any references in documentation, CI/CD pipelines, submodules, scripts, and bookmarks.

Old links under:

```
github.com/sk-ai-net/...
```

should now use:

```
github.com/SKaiNET-developers/...
```

---

## 💬 Questions?

If you encounter any issues or broken links, please open an issue in the relevant repository under **SKaiNET-developers**.

Thanks for your support and happy hacking 🚀
