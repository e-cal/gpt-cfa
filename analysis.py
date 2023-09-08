from collections import defaultdict
import pandas as pd
import glob

ATTMEPTS_PATH = "attempts/"

ANALYZE_GLOBS = [
    # "l1/chatgpt*",
    # "l1/gpt4*",
    # "l1/cot/chatgpt*",
    # "l1/cot/gpt4*",
    # "l2/chatgpt*",
    # "l2/gpt4*",
    "l2/cot/chatgpt*",
    "l2/cot/gpt4*",
    # "l1/cotam/*",
]


def l1eval(df, scores, spacing):
    for chapter in df["chapter_name"].unique():
        df_chapter = df[df["chapter_name"] == chapter]
        num_correct = sum(df_chapter["correct"] == "yes")
        score = num_correct / len(df_chapter)
        scores[chapter].append(score)
        spacer = " " * (spacing - len(chapter))

        print(
            f"  {chapter}:{spacer}{num_correct:>2}/{len(df_chapter):<2}  {score*100:.2f}%"
        )

def l2eval(df, scores, spacing):
    for chapter in df["chapter_name"].unique():
        df_chapter = df[df["chapter_name"] == chapter]
        num_correct = sum(df_chapter["correct"] == "yes")
        score = num_correct / len(df_chapter)
        scores[chapter].append(score)
        spacer = " " * (spacing - len(chapter))

        print(
            f"  {chapter}:{spacer}{num_correct:>2}/{len(df_chapter):<2}  {score*100:.2f}%"
        )


for glob_pattern in map(lambda x: ATTMEPTS_PATH + x, ANALYZE_GLOBS):
    print(f"\n\n{glob_pattern:^30}\n")
    
    scores = defaultdict(list)
    files = sorted(
        glob.glob(glob_pattern),
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )
    glob_spacing = 0
    for i, fp in enumerate(files):
        print(fp, end=": ")

        df = pd.read_csv(fp)

        if "l2" in fp:
            l2_df = pd.read_json(f"data/l2/cfa_level_2_exam_{i+1}.json")
            inc = 0
            for k, row in l2_df.iterrows():
                for j, q in enumerate(row["cfa2_cbt_questions"]):
                    # df row j chapter_name
                    df.loc[j+inc, "chapter_name"] = q["chapter_name"]
                inc += j+1  # type: ignore
        # print a row where chaptername is nan
        spacing = max(map(len, df["chapter_name"].unique())) + 4
        if spacing > glob_spacing: glob_spacing = spacing

        spacer = " " * (spacing - len(fp) + 1)
        num_correct = sum(df["correct"] == "yes")
        score = num_correct / len(df)
        print(f"{spacer}{num_correct}/{len(df)}  {score*100:.2f}%")

        scores["overall"].append(score)

        if "l1" in fp: l1eval(df, scores, spacing)
        else: l2eval(df, scores, spacing)

        print()

    # average score
    for key, kscores in scores.items():
        avg = sum(kscores) / len(kscores)
        var = sum((s - avg)**2 for s in kscores) / (len(kscores))
        print(f"{key:<{glob_spacing}} {avg*100:.2f}% ± {var*100:.2f}%")
