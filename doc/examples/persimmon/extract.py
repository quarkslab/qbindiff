#!/bin/env python

import os, json, requests

BASE_URL = "http://kaki.persimmon.qb:8080/v0"
ARCH = "x64"
KB_TARGET = "5013941"
BINARY = "ntoskrnl.exe"


def json_req(url, params=None):
    if url[0] != "/":
        url = "/" + url
    r = requests.get(BASE_URL + url, params=params)
    return r.json()


def get_patch(updateKb, arch):
    patches = json_req(f"updates/{updateKb}/patches")
    for patch in patches["Content"]:
        if patch["Product"]["Arch"] == arch:
            return patch["Id"]


def get_file(patchId, name):
    patches = json_req(f"patches/{patchId}/files", {"name": name})
    return patches["Content"][0]


def dump_file(fileId, isPdb):
    print("[+] Downloading file")
    r = requests.get(BASE_URL + f"/files/{fileId}", params={"download": "file"})
    rawFile = r.content
    rawPdb = None
    if isPdb:
        print("[+] Downloading pdb")
        r = requests.get(BASE_URL + f"/files/{fileId}", params={"download": "pdb"})
        rawPdb = r.content
    return (rawFile, rawPdb)


def write(filename, binary):
    try:
        os.mkdir("output")
    except:
        pass

    with open(f"output/{filename}", "wb") as f:
        f.write(binary)


if __name__ == "__main__":
    prevKb = json_req(f"updates/{KB_TARGET}")["PreviousKb"][0]

    # Get patches
    patchTarget = get_patch(KB_TARGET, ARCH)
    patchPrev = get_patch(prevKb, ARCH)
    # ~ print(KB_TARGET, patchTarget)
    # ~ print(prevKb, patchPrev)

    # Get files
    targetFile = get_file(patchTarget, BINARY)
    prevFile = get_file(patchPrev, BINARY)
    # ~ print(targetFile)
    # ~ print(prevFile)

    binary, pdb = dump_file(targetFile["FileId"], targetFile["IsPdbPresent"])
    write(f"{KB_TARGET}-{BINARY}", binary)
    if pdb:
        write(f"{KB_TARGET}-{BINARY}.pdb", pdb)

    binary, pdb = dump_file(prevFile["FileId"], prevFile["IsPdbPresent"])
    write(f"{prevKb}-{BINARY}", binary)
    if pdb:
        write(f"{prevKb}-{BINARY}.pdb", pdb)
