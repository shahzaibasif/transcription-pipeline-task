import json,os
def save_output(result):
    os.makedirs("output",exist_ok=True)
    open("output/transcript.txt","w",encoding="utf-8").write(result["text"])
    json.dump(result,open("output/transcript.json","w",encoding="utf-8"),indent=4)
