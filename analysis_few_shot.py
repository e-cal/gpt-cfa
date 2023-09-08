from collections import defaultdict
import pandas as pd
import glob

ATTMEPTS_PATH = "attempts/"

ANALYZE_GLOBS = [
    "few_shot/*"
]


def l1eval(df, scores, spacing):
    for chapter in df["chapter_name"].unique():
        df_chapter = df[df["chapter_name"] == chapter]
        num_correct = sum(df_chapter["correct"] == "yes")
        score = num_correct / len(df_chapter)
        scores[chapter].append(score)
        spacer = " " * (spacing - len(chapter))
        print(
            f"  {chapter}:{spacer}{num_correct:>2}/{len(df_chapter):<2}  {score * 100:.2f}%"
        )


def l2eval(df, scores, spacing):
    for chapter in df["chapter_name"].unique():
        df_chapter = df[df["chapter_name"] == chapter]
        num_correct = sum(df_chapter["correct"] == "yes")
        score = num_correct / len(df_chapter)
        scores[chapter].append(score)
        spacer = " " * (spacing - len(chapter))

        print(
            f"  {chapter}:{spacer}{num_correct:>2}/{len(df_chapter):<2}  {score * 100:.2f}%"
        )


for glob_pattern in map(lambda x: ATTMEPTS_PATH + x, ANALYZE_GLOBS):
    print(f"\n\n{glob_pattern:^30}\n")

    scores = defaultdict(list)
    all_files = glob.glob(glob_pattern)

    # get fsr files
    fsr_files = [file for file in all_files if "fsr" in file]

    fsr_2_shot_files = [file for file in fsr_files if "2_shots" in file]
    fsr_2_shot_files = sorted(
        fsr_2_shot_files,
        key=lambda x: int(x.split("_")[3]),
    )

    fsr_4_shot_files = [file for file in fsr_files if "4_shots" in file]
    fsr_4_shot_files = sorted(
        fsr_4_shot_files,
        key=lambda x: int(x.split("_")[3]),
    )

    fsr_6_shot_files = [file for file in fsr_files if "6_shots" in file]
    fsr_6_shot_files = sorted(
        fsr_6_shot_files,
        key=lambda x: int(x.split("_")[3]),
    )

    # get fst files
    fst_files = [file for file in all_files if "fst" in file]
    fst_files = sorted(
        fst_files,
        key=lambda x: int(x.split("_")[3]),
    )

    # create dict with file paths
    file_dict = {
        "FSR_2_SHOTS": fsr_2_shot_files,
        "FSR_4_SHOTS": fsr_4_shot_files,
        "FSR_6_SHOTS": fsr_6_shot_files,
        "FST_1_SHOT": fst_files,
    }

    glob_spacing = 0

    for file_type, files in file_dict.items():
        print(f"=========={file_type}==============")
        for i, fp in enumerate(files):
            print(fp, end=": ")

            df = pd.read_csv(fp)

            # print a row where chaptername is nan
            spacing = max(map(len, df["chapter_name"].unique())) + 4
            if spacing > glob_spacing: glob_spacing = spacing

            spacer = " " * (spacing - len(fp) + 1)
            num_correct = sum(df["correct"] == "yes")
            score = num_correct / len(df)
            print(f"{spacer}{num_correct}/{len(df)}  {score * 100:.2f}%")

            scores["overall"].append(score)

            l1eval(df, scores, spacing)
            print()

        # average score
        for key, kscores in scores.items():
            avg = sum(kscores) / len(kscores)
            var = sum((s - avg) ** 2 for s in kscores) / (len(kscores))
            print(f"{key:<{glob_spacing}} {avg * 100:.2f}% ± {var * 100:.2f}%")
