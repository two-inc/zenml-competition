import numpy as np
import streamlit as st
from zenml.post_execution import get_pipeline

from src.run_training_pipeline import run_training_pipeline
from src.util.data_access import load_data


# export PYTHONPATH=/home/jamesworthington/zenml-competition
# ssh -L 8000:127.0.0.1:8501 jworthington.europe-west2-a.zenml-competition
def main():
    """Main function for streamlit"""

    # df = load_data()
    # categories = list(df['category'].unique())
    # categories = [
    #    "es_transportation",
    #    "es_health",
    #    "es_otherservices",
    #    "es_food",
    #    "es_hotelservices",
    #    "es_barsandrestaurants",
    #    "es_tech",
    #    "es_sportsandtoys",
    #    "es_wellnessandbeauty",
    #    "es_hyper",
    #    "es_fashion",
    #    "es_home",
    #    "es_contents",
    #    "es_travel",
    #    "es_leisure",
    # ]

    st.title("Detecting Fraudulent Financial Transactions")

    st.markdown("<Problem statement>")

    st.markdown(
        """
    - Order Amount:
    - Category:
    - Gender: Male, Female, Enterprise, Unknown
    - Step: TBC
    """
    )

    # amount = st.number_input("Order Amount")
    # category = st.selectbox("Category", categories)
    # gender = st.selectbox("Gender", ["M", "F", "E", "U"])
    # step = st.number_input("Step", 0, 179)

    if st.button("Predict"):
        # service = load_last_service_from_step(
        #     pipeline_name="continuous_deployment_pipeline",
        #     step_name="model_deployer",
        #     running=True,
        # )

        # put existing pipeline in here that is already instantiated
        # run_training_pipeline()
        p = get_pipeline(pipeline_name="train_pipeline")
        last_run = p.runs[-1]
        trainer_step = last_run.get_step("trainer")
        model = trainer_step.output.read()

        data = [1, 1, 1, 1, 1, 1, 1, 1, 1]

        data = np.array(data).reshape(1, -1)

        # data = [
        #     amount,
        #     category,
        #     gender,
        #     step
        # ]

        # data.reshape(1,-1)

        # # prediction = service.predict()
        prediction = model.predict(data)
        st.success(
            f"Given the customer's historical data, model says LEGITIMATE {prediction}"
        )

        # # prediction = True
        # # if prediction:
        # #     st.success("Given the customer's historical data, model says LEGITIMATE")
        # # else:
        # #     st.error("Given the customer's historical data, model says FRAUDULENT")


if __name__ == "__main__":
    main()
