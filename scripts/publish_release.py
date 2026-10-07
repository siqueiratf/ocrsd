import argparse
import json
import mimetypes
import os
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request


def release_notes(tag: str, changelog: Path) -> str:
    text = changelog.read_text(encoding="utf-8")
    marker = f"## [{tag}]"
    if marker not in text:
        raise ValueError(f"Changelog sem entrada para {tag}")
    section = text.split(marker, 1)[1].split("\n## ", 1)[0]
    return section.partition("\n")[2].strip() + "\n"


def request_json(url: str, token: str, *, method: str = "GET",
                 payload: dict | None = None, data: bytes | None = None,
                 content_type: str = "application/json"):
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": content_type,
        "X-GitHub-Api-Version": "2022-11-28",
    })
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Publica um release com notas do changelog e arquivos validados.")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    parser.add_argument("--assets", type=Path, nargs="+", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    notes = release_notes(args.tag, args.changelog)
    for path in args.assets:
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Arquivo ausente ou vazio: {path}")
    if args.dry_run:
        print(json.dumps({"tag": args.tag, "body": notes,
                          "assets": [path.name for path in args.assets]}, ensure_ascii=False, indent=2))
        return
    token = os.environ["GITHUB_TOKEN"]
    repository = os.environ["GITHUB_REPOSITORY"]
    endpoint = f"https://api.github.com/repos/{repository}/releases"
    try:
        release = request_json(f"{endpoint}/tags/{urllib.parse.quote(args.tag, safe='')}", token)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        release = request_json(endpoint, token, method="POST", payload={
            "tag_name": args.tag, "name": args.tag, "body": notes,
            "draft": True, "prerelease": False,
        })
    upload_url = release["upload_url"].partition("{")[0]
    assets = {item["name"]: item for item in release.get("assets", [])}
    for path in args.assets:
        if path.name in assets:
            continue
        assets[path.name] = request_json(
            upload_url + "?" + urllib.parse.urlencode({"name": path.name}),
            token, method="POST", data=path.read_bytes(),
            content_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        )
    if release.get("draft"):
        release = request_json(f"{endpoint}/{release['id']}", token, method="PATCH",
                               payload={"draft": False, "make_latest": "true"})
    report = {
        "tag": args.tag, "id": release["id"], "url": release["html_url"],
        "published_at": release["published_at"],
        "assets": [{"name": item["name"], "size": item["size"],
                    "url": item["browser_download_url"]} for item in assets.values()],
    }
    Path("dist/release.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(f"Release publicado: [{args.tag}]({report['url']})\n")
    print(report["url"])


if __name__ == "__main__":
    main()
