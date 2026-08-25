# Server Change Log

## Repository

- Project path: `/home/23adr188/chest_xray_project`
- Remote: `https://github.com/Sindhuja-Raja/medical_final_year.git`
- Branch: `master`
- Remote tracking: `origin/master`
- Time zone: `UTC+05:30` (server local time)
- Log updated: `2026-08-25 21:12:24 +05:30`

## Server Timing

### Simple Explanation

This project was already being prepared on the server by **12 August 2026 at 3:05:59 PM**. That is the earliest time recorded for any project file. The project work continued through **25 August 2026**, with the model files last updated shortly after noon.

At **9:08:46 PM on 25 August 2026**, the complete project was collected into its first Git commit. This created a permanent snapshot containing the source code, scripts, documentation, configuration, and tests.

After that snapshot was created, the server connected the project to the requested GitHub repository and uploaded the `master` branch. The final change-log commit was completed at **9:12:47 PM**, and GitHub confirmed that the branch was uploaded successfully. The server had no uncommitted changes afterward.

In short: **the earliest recorded server IN TIME was 12 August at 3:05:59 PM, and the final recorded OUT TIME was 25 August at 9:12:47 PM.** This is the observed time window for the project files and publishing work, not a claim that no activity occurred before the first file timestamp.

| Timing | Date and time | Evidence and meaning |
| --- | --- | --- |
| IN TIME | 2026-08-12 15:05:59 +05:30 | Earliest project file timestamp found on the server: `MASTER_PROJECT_PLAN.md` |
| DEVELOPMENT CHECKPOINT | 2026-08-25 12:03:29-12:03:37 +05:30 | Model files under `src/models/` were last modified on the server |
| REPOSITORY IN TIME | 2026-08-25 21:08:46 +05:30 | Initial Git commit created: `b46930d` |
| LOG OUT TIME | 2026-08-25 21:12:47 +05:30 | Change-log commit created and the updated `master` branch was pushed successfully |

### Detailed Server Activity Window

- **Server/project IN TIME:** `2026-08-12 15:05:59 +05:30`. This is the earliest filesystem timestamp available for the project contents.
- **Project preparation period:** Project files, documentation, scripts, source code, and tests were present and modified during the period ending with the model-file timestamps on `2026-08-25 12:03:37 +05:30`.
- **Repository capture:** `2026-08-25 21:08:46 +05:30`. The complete project state was committed as the initial commit.
- **Remote configuration and publication:** After the initial commit, the `origin` remote was configured with the requested GitHub URL and `master` was pushed.
- **Server/log OUT TIME:** `2026-08-25 21:12:47 +05:30`. The log file was committed and the push command completed with `master -> master`.
- **Total observed server activity window:** 13 days, 6 hours, 6 minutes, 48 seconds, from the earliest available project timestamp to the final published log commit.

The filesystem timestamps show when project files were present or modified on this server. They do not prove that no work happened before the earliest available timestamp. The exact HTTPS network-transfer start time is not stored by Git; the final successful push is therefore recorded at the completion point shown above.

## Changes From Start

| Date and time | Change | Result |
| --- | --- | --- |
| 2026-08-25 21:08:46 +05:30 | Created the initial Git commit, `b46930d` | 48 files added, 8,600 lines inserted |
| 2026-08-25 after 21:08:46 +05:30 | Added the GitHub remote as `origin` | Remote configured successfully |
| 2026-08-25 21:12:47 +05:30 | Pushed the updated `master` branch to GitHub | `master` published and set to track `origin/master` |

## Initial Commit Contents

The initial commit included:

- Project configuration files in `configs/`
- Documentation and manuscript files in `docs/`
- Dataset, validation, analysis, and training scripts in `scripts/`
- Model, training, and evaluation code in `src/`
- Tests in `tests/`
- Root project files including `.gitignore`, `README.md`, `AGENTS.md`, `MASTER_PROJECT_PLAN.md`, and `requirements.txt`

Commit summary:

```text
Commit: b46930d664aabd6acb271016e86aed66befbefd7
Message: Initial project commit
Files changed: 48
Insertions: 8,600
Deletions: 0
Author: Vipin <23adr188@master.intelunnati>
```

## Final Server State

The working tree was clean after the push. The local branch reported:

```text
master...origin/master
```

Git records the commit time and branch synchronization, but it does not record the exact time when the HTTPS push completed. Therefore, the push entry above is recorded as occurring after the initial commit, based on the server command sequence, without inventing an exact timestamp.

## Server Login and Logout Records

These are operating-system login sessions recorded by the server's `wtmp` database. They are different from Git commits. Each row represents one terminal login session.

| User | Terminal | Login IN TIME | Logout OUT TIME | Duration | Source IP | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `23adr188` | `pts/3` | 2026-08-25 21:44:29 | Not recorded yet | Active | `10.1.37.48` | Still logged in |
| `23adr188` | `pts/1` | 2026-08-25 20:50:55 | 2026-08-25 20:52:54 | 1 minute | `10.1.82.61` | Logged out |
| `23adr188` | `pts/3` | 2026-08-25 19:35:25 | 2026-08-25 19:44:32 | 9 minutes | `10.1.82.61` | Logged out |
| `23adr188` | `pts/1` | 2026-08-25 19:18:33 | 2026-08-25 19:20:56 | 2 minutes | `10.1.82.61` | Logged out |
| `23adr188` | `pts/1` | 2026-08-25 19:14:32 | 2026-08-25 19:17:34 | 3 minutes | `10.1.82.61` | Logged out |
| `23adr188` | `pts/1` | 2026-08-25 17:18:11 | 2026-08-25 17:20:56 | 2 minutes | `10.1.82.61` | Logged out |
| `23alr009` | `pts/1` | 2026-08-25 14:40:26 | 2026-08-25 14:57:32 | 17 minutes | `10.1.110.143` | Logged out |
| `23alr009` | `pts/1` | 2026-08-25 12:06:32 | 2026-08-25 12:30:27 | 23 minutes | `10.2.11.25` | Logged out |
| `kec3155` | `pts/0` | 2026-08-25 11:44:35 | Not recorded yet | Active | `10.2.8.153` | Still logged in |
| `23aur005` | `pts/0` | 2026-08-25 11:35:35 | 2026-08-25 11:44:12 | 8 minutes | `10.2.8.153` | Logged out |
| `23adr169` | `pts/1` | 2026-08-25 10:55:46 | 2026-08-25 11:35:37 | 39 minutes | `10.1.111.74` | Logged out |
| `23adr169` | `pts/1` | 2026-08-25 10:39:24 | 2026-08-25 10:55:04 | 15 minutes | `10.1.111.74` | Logged out |

### How To Get Every Record

Run this command on the server to print the complete available login/logout history, including all users and source addresses:

```bash
last -Fai
```

Use `lastlog` to see the latest login for every local account:

```bash
lastlog
```

The server currently has no application-level login system in this project. Therefore, these records are Linux terminal/SSH session records. A session shown as `still logged in` has an IN TIME but will receive its OUT TIME only after that session exits. The login database does not provide records for sessions that occurred before its retained history.