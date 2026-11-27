# AGENTS.md: Strategy Lab (handoff + entry point)

**Written:** 2026-10-12, on the user's first computer, where **nothing was installed or built**.
**Who reads this:** the AI agent (or human) that implements Strategy Lab on **another computer**.
**Read this whole file before doing anything.** It records what the user expects, every decision made so far, how the user likes to work, and exactly where to start.

---

## 0. Read order

1. **This file**: intent, decisions, working rules, setup, first steps.
2. **Master plan**: `Game_Theory_Practice_Web_App_Plan.md` (v3). Canonical copy: `<MATLAB repo>/Planning/Game Theory/`. If the repo is not on this computer, the user copies it next to this file as `docs/MASTER_PLAN.md`. It covers architecture, the primitive library, insight outputs, finance indicators, the cloud stack, phases P0–P11, the status board (§13) and the agent rules (§12).
3. **Phase plan P0**: `docs/phases/P0_Contracts.md`. This is the first thing to implement, after approval.

If the master plan or the P0 plan is missing, **stop and ask the user**. Do not rebuild them from memory.

---

## 1. What the user wants (intent, in plain words)

The user is studying game theory (Fudenberg & Tirole, *Game Theory*, Ch. 1) and wants a **web app for practising game theory applied to economics and other fields**.

- **Practice tool, not an answer machine.** The user builds or opens a model, **predicts** a result, then checks it against the engine, with hints. The engine explains *why* (certificates, deviations, failed conditions).
- **Built from economic functions.** Models are composed from **demand, supply/cost, price/market rules, capacity, uncertainty and return (profit) functions**. The engine derives each player's payoff from these and **extracts useful information**: equilibria, best responses, regime maps over parameters, comparative statics, welfare, markups, financial indicators, and plain-language summaries.
- **The textbook exercises are a reference only.** Exercises 1.1–1.7 (2×2 characterization, voting game, Nash demand game / Binmore, Edgeworth duopoly, final-offer arbitration, a 3×3 unique-equilibrium game, the Groves public-good mechanism) were used to learn *what the app must be able to do* (master plan §2). **The user explicitly does NOT want these exercises solved.** Never write their solutions into the app, docs or tests.
- **High-performance engine.** The user asked for C/C++/Rust. Decision: **Rust**, compiled to **WebAssembly** for the browser and native for the cloud (same crate).
- **Modern, professional UI/UX.** React + TypeScript, Tailwind, shadcn/ui, KaTeX, D3/visx, light/dark themes, accessible (WCAG 2.2 AA). Details are in master plan §6.2.
- **Production-style cloud system on Google Cloud.** The user has a **$300 free-trial credit** and wants to use it. Results are **uploaded to the cloud**. Redundancy does not matter (it's for practice).
- **Follow the conventions of** `C:\Ocean Library\Production SI\AGENTS.md`: plan per phase, contracts before code, component → language → tool table with reasons, "done when" tests, status board, explain commands. Those conventions are already folded into master plan §6.5 and §12.
- **Use what the industry uses today.** The user has Cassandra, Redis, Postgres and others available and asked the agent to choose. The choices and reasons are below.

---

## 2. Decision log (all dated 2026-10-12 unless stated)

| # | Decision | Detail |
|---|---|---|
| D1 | Engine language | Rust → WASM (browser) + native (Cloud Run). Not C++ (WASM toolchain and safety). |
| D2 | Supply/demand source | Use **the user's formula or data first**; also "derived" from a population distribution. **Approximation from partial information is later (P7)**, and until then the app asks for input rather than guessing. Every result is labelled with its source (`provided / data / derived / approximated`). |
| D3 | "Support" in the user's request | Meant **supply** (the user confirmed). Mixed-strategy support diagnostics also exist, separately. |
| D4 | Returns | **Simple profit** (p·q − C(q) ± transfers) is what players optimise. **Financial indicators** are computed from outcomes: margins, markup/Lerner, break-even, ROI, utilisation (P5); NPV, IRR, payback, profitability index, NPV risk (P6). |
| D5 | Scope | **All phases** (the user said "End" = go to the end), not just a first version. |
| D6 | Location | `<MATLAB repo>/+OceanViz/+App/StrategyLab/`. The folder name has no `+`, so MATLAB ignores it, and `genpath` never enters `+` folders. If the MATLAB repo is not on the new computer, **ask the user** where to put it. |
| D7 | Storage | **Accounts + cloud sync on Google Cloud** (not local-only). IndexedDB only as an offline draft cache. |
| D8 | Cloud stack | Firebase Hosting + Firebase Auth (Google sign-in); **Rust axum API on Cloud Run**; **Cloud SQL PostgreSQL**; **Cloud Storage** for results (content-addressed by `model_hash + engine_version`); **Pub/Sub → Cloud Run Jobs** for sweeps; Secret Manager; Terraform; Cloud Build + Artifact Registry; OpenTelemetry → Cloud Logging/Monitoring/Trace. |
| D9 | Redis | **Deferred to P11**, only if measurements show a need. The managed version (Memorystore) has a fixed monthly cost. |
| D10 | Cassandra | **Not used.** It's built for huge write volume across many machines; this app's data is small and relational. |
| D11 | Database cost plan | **A → B:** Cloud SQL during the trial (paid by the credit); before the credit ends, move Postgres to the Always-Free e2-micro VM (self-managed) or tear it down. Only the connection string changes. |
| D12 | Trial window | Account registered **2026-09-26**, so the credit ends around **2026-12-25** (the user should confirm in the Billing console). Finish cloud phases P8–P9 by **early December 2026**. GPUs are blocked on trial accounts (not needed). Cloud SQL has no Always-Free tier. |
| D13 | Phase order | P0 → P4 → P8 → P9 → P5 → P6 → P10 → P7 → P11, so the credit window is used on a working app. |
| D14 | Implementation machine | **A Fedora Linux computer**, not the first (Windows) computer. Everything is built there, driven by the agent (§4). |

---

## 3. How the user wants agents to work

1. **Plan first, code only on approval.** Before each phase, write `docs/phases/Pn_<name>.md` and wait for an explicit "yes". A question from the user is not a request to build.
2. **One phase per session.** Finish, test, update the status board (master plan §13), stop.
3. **Never push to GitHub. No commits or PRs unless the user explicitly asks.**
4. **No cloud action without explicit approval each time**: `terraform apply`, deploys, anything billable. State what it creates and its approximate cost first.
5. **No hardcoded or mock data presented as real.** Test fixtures are generated or are clearly *inputs*; never hand-typed "results".
6. **Maths from the textbook** (Fudenberg & Tirole; standard corporate-finance texts for indicators). Cite in code comments. **Announce every non-textbook addition** to the user (e.g. Lemke–Howson, RFC 8785 hashing, Gambit as an oracle).
7. **Explain commands.** The user is learning. For every terminal command you ask them to run, explain each part (tool, subcommand, flags, arguments) and what the output means. Prefer having the user run installs themselves, with explanation.
8. **Verify versions and GCP prices at implementation time.** Nothing is pinned in the plans.
9. **Contracts before code**: anything crossing browser ↔ API ↔ worker is defined in the generated schema first.
10. **Secrets never in files or images.**
11. **Keep token use low**: narrow sessions, targeted tests, terse progress updates; spawn sub-agents only when asked.
12. **Reports go in the chat.** Create extra markdown files only when the user asks (phase plans and this file are expected).
13. **Use they/them** for the user unless told otherwise.

---

## 4. Setting up the new computer

### 4.0 Target machine: Fedora Linux (decided 2026-10-12)

The implementation computer runs **Fedora** (the user's machines run Fedora 44). Confirm which machine with the user:
- **Dell XPS laptop** (`laptop`): 32 GB RAM, 1 TB NVMe. **Recommended**, with plenty of room for Rust builds + Docker.
- **Lab desktop** (`lab-desktop`): 16 GB RAM, but only a 240 GB system SSD that is shared with KVM VMs (Kali). Workable, but watch the disk.

The user is learning Linux and security, so explain every command part by part (tool, subcommand, flags, arguments) and what its output means.

### 4.1 First, check what exists
Run these and report the results to the user before installing anything:

```bash
cat /etc/os-release
```
```bash
df -h ~
```
```bash
free -h
```
```bash
node --version; npm --version; pnpm --version; rustup show; cargo --version; gcc --version; docker --version; podman --version; git --version
```
Missing tools print "command not found". That is expected and is the list of what to install.

### 4.2 Install what's missing: guide the user (Fedora commands; explain the syntax)

| Tool | Needed from | Fedora command | What to explain |
|---|---|---|---|
| C toolchain | P0 | `sudo dnf install gcc gcc-c++ make pkgconf-pkg-config openssl-devel` | `sudo` = run as admin; `dnf` = Fedora package manager; `install` + package names. Rust needs a C linker (`cc`). `openssl-devel`/`pkgconf` are needed by some Rust crates (TLS, Postgres). **No Visual Studio on Linux.** |
| Rust | P0 | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` (official rustup method), then `source ~/.cargo/env` or open a new terminal | `curl` downloads; `--proto '=https'` allows HTTPS only; `--tlsv1.2` sets a minimum TLS version; `-sSf` = silent, but show errors, and fail on HTTP errors; `\| sh` pipes the script into the shell. Security note for this user: piping a script into the shell means trusting rustup.rs. The alternative is `sudo dnf install rustup` then `rustup-init`. `rustup show` should print `x86_64-unknown-linux-gnu`. Optional: `rustup component remove rust-docs`. |
| WASM target | P0 | `rustup target add wasm32-unknown-unknown` | arch-vendor-OS triple; "unknown OS" = the browser sandbox |
| Node.js | P0 | `sudo dnf install nodejs` (check the version is current LTS or newer; otherwise use `fnm`/`nvm`) | Fedora's packaged Node |
| pnpm | P0 | `npm install -g pnpm` **without sudo**: first set a user prefix: `npm config set prefix ~/.local` (and make sure `~/.local/bin` is on `PATH`) | Avoid `sudo npm -g` (it writes root-owned files into system folders) |
| wasm-pack / wasm-bindgen-cli | P0 (after `Cargo.lock` exists) | `cargo install …` with the version matching the locked `wasm-bindgen` crate | Check wasm-pack is still maintained; otherwise use wasm-bindgen-cli + a build script |
| Playwright Chromium | P0 | `pnpm exec playwright install chromium`; if system libraries are missing, Playwright prints the `dnf` packages to install | Chromium only (~150–300 MB). Playwright officially targets Ubuntu/Debian; on Fedora, install the missing libraries manually or run the tests inside the official Playwright Docker image. |
| Containers | P8 | Fedora ships **Podman** (Docker-compatible, rootless). Either use Podman + `podman compose`, or install Docker Engine from Docker's Fedora repository. Decide in the P8 plan. | **SELinux** is on by default: bind-mounted volumes need the `:Z` suffix (e.g. `-v ./data:/data:Z`), otherwise you get "permission denied". |
| gcloud | P8 | Google's official `dnf` repository for the Google Cloud CLI (follow the current docs) | Explain repo files (`/etc/yum.repos.d/`) and GPG key checks |
| Terraform | P8 | HashiCorp's official `dnf` repository | same |
| Firebase CLI | P8 | `npm install -g firebase-tools` (user prefix, no sudo) | |

### 4.3 Linux-specific notes
- **firewalld** is active (default zone `public`). Dev servers (Vite, the local API) should listen on `localhost` only, so **no firewall ports need opening**. If the user wants to open the app from another device, ask first, and open the port only in a trusted zone, with an explanation.
- **Files copied from Windows:** add a `.gitattributes` with `* text=auto eol=lf` and check that shell scripts have LF line endings. CRLF breaks `#!/bin/sh` scripts.
- **The file system is case-sensitive**: import paths must match file-name case exactly.
- **Paths with `+`** (`+OceanViz/+App/StrategyLab`) are fine on Linux. If the MATLAB repo is not copied, use a plain folder such as `~/OceanLibrary/StrategyLab` (ask the user, D6).
- **Rebooting:** on Fedora desktop, `sudo reboot` can be refused by the GNOME session inhibitor. Use the GUI Restart or `sudo systemctl reboot -i`.

### 4.4 Disk estimate (tell the user)
- P0: ~2–4 GB (Rust ~1–1.5 GB, gcc toolchain a few hundred MB, Node packages + Chromium ~0.5–1 GB). This is much less than on Windows, because Linux needs no Visual Studio.
- By P8: ~8–15 GB in total. Rust `target/` grows to 1–5 GB (`cargo clean` frees it); container images ~0.5–1 GB.

---

## 5. Where to start (first session on the new computer)

1. Confirm with the user where the project folder lives (D6) and that the master plan + P0 plan are present.
2. Do the environment check (§4.1) and guide the installs (§4.2).
3. Ask the two open P0 questions (P0 plan §7):
   - (a) the prerequisites are installed (or the user approves you guiding the rest);
   - (b) the starter scenarios: **2×2 matrix game, linear-demand Cournot duopoly, Bertrand–Edgeworth duopoly with capacity**. These are model *inputs* only.
4. On approval, implement P0 exactly as in its plan. "Done" = `pnpm check` green:
   - `fmt`/`clippy`/`test`;
   - schema round-trip;
   - rejection tests;
   - generated-files drift check;
   - hash stability;
   - WASM ↔ native parity;
   - a Playwright smoke test.
5. Update the status board (master plan §13), report in chat, stop. The next session writes the P1 plan.

---

## 6. Open questions (ask when the relevant phase starts)

| Phase | Question |
|---|---|
| P8 | One cloud environment only, or also staging? Staging doubles the Cloud SQL cost. |
| P9 | How long to keep sweep results in Cloud Storage (lifecycle rule). |
| P11 | Whether Redis / BigQuery are justified by measurements. |
| Any | If the project is not inside the MATLAB repo on the new computer, where it should live and whether the optional `+OceanAgents` / `+OceanData` links still matter (they are optional and not part of the build). |

---

## 7. Glossary of the user's wording

- "support" → **supply** function (D3).
- "return" → **simple profit**, plus **financial indicators** (D4).
- "End" (on scope) → build **all phases** (D5).
- "reddits" → **Redis** (D9).
- "upload results on cloud, don't worry about redundancy" → single-region Cloud Storage, no replication, lifecycle cleanup.
