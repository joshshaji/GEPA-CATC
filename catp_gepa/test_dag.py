

from catp_gepa.metric import dag_loss, plan_to_dag


if __name__ == "__main__":

    plan_1_dag = plan_to_dag([
                "image_denoising",
                ["input_of_query"],
                "image_deblurring",
                ["image_denoising"],
            ])

    gold_dag = plan_to_dag([
                "image_deblurring",
                ["input_of_query"],
                "image_denoising",
                ["image_denoising"],
            ])

    plan_loss = dag_loss(plan_1_dag, gold_dag)
    print("dag_loss:", plan_loss)
