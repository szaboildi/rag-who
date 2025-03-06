
import pandas as pd
import time
import os



def export_qa_lists(queries:list[str], responses:list[str],
                    model:str, temperature:float, export_folder:str):
    qa_df = pd.DataFrame({"Query": queries, "Response": responses})

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    qa_df.to_csv(
        os.path.join(
            export_folder, f"QA_{model}_temp{temperature}_{timestamp}.csv"),
        index=False)

    return
