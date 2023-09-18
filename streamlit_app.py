import datetime
import json
import os
import requests
import urllib

import fasteners
import pandas as pd
import streamlit as st


class ChatHistoryRecord:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        lockfile = f"{storage_path}.rwlock"
        self.lock = fasteners.InterProcessLock(lockfile)

    def _unsafe_read(self) -> pd.DataFrame:
        try:
            data = pd.read_parquet(self.storage_path)
        except FileNotFoundError:
            return pd.DataFrame()
        else:
            return data

    def read(self) -> pd.DataFrame:
        with self.lock:
            return self._unsafe_read()

    def _unsafe_write(self, data: pd.DataFrame):
        data.to_parquet(self.storage_path)

    def write(self, data: pd.DataFrame):
        with self.lock:
            return self._unsafe_write(data)

    def append(self, record: dict) -> pd.DataFrame:
        """Sequential write single record."""
        new_df = pd.DataFrame.from_dict(record, orient="index").T.set_index("timestamp")
        with self.lock:
            data = self._unsafe_read()
            new_data = pd.concat([data, new_df])
            self._unsafe_write(new_data)

        return new_data


@st.cache_resource
def get_chat_record():
    return ChatHistoryRecord("data/persist/chathistory.parquet")


# Don't change this. It is enforced server-side too.
MAX_PREDICTION_FILE_SIZE_BYTES = 52428800  # 50 MB


class DataRobotPredictionError(Exception):
    """Raised if there are issues getting predictions from DataRobot"""


def make_datarobot_deployment_unstructured_predictions(
    endpoint,
    data,
    deployment_id,
    token,
    key,
    mimetype="application/json",
    charset="UTF-8",
):
    """
    Make unstructured predictions on data provided using DataRobot deployment_id provided.
    See docs for details:
         https://staging.datarobot.com/docs/predictions/api/dr-predapi.html

    Parameters
    ----------
    data : bytes
        Bytes data read from provided file.
    deployment_id : str
        The ID of the deployment to make predictions with.
    mimetype : str
        Mimetype describing data being sent.
        If mimetype starts with 'text/' or equal to 'application/json',
        data will be decoded with provided or default(UTF-8) charset
        and passed into the 'score_unstructured' hook implemented in
        custom.py provided with the model.

        In case of other mimetype values data is treated as binary
        and passed without decoding.
    charset : str
        Charset should match the contents of the file, if file is text.

    Returns
    -------
    data : bytes
        Arbitrary data returned by unstructured model.

    Raises
    ------
    DataRobotPredictionError if there are issues getting predictions from DataRobot
    """
    # Set HTTP headers. The charset should match the contents of the file.
    headers = {
        "Content-Type": "{};charset={}".format(mimetype, charset),
        "Authorization": "Bearer {}".format(token),
        "DataRobot-Key": key,
    }

    url = endpoint
    url = f"{endpoint}/predApi/v1.0/deployments/{deployment_id}/predictionsUnstructured"

    # Make API request for predictions
    predictions_response = requests.post(
        url,
        data=json.dumps(data),
        headers=headers,
    )
    _raise_dataroboterror_for_status(predictions_response)
    # Return raw response content
    return predictions_response.json()


def _raise_dataroboterror_for_status(response):
    """Raise DataRobotPredictionError if the request fails along with the response returned"""
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        err_msg = "{code} Error: {msg}".format(
            code=response.status_code, msg=response.text
        )
        raise DataRobotPredictionError(err_msg)


def make_prediction(prompt, history=None) -> dict:
    endpoint = st.session_state.datarobot_prediction_endpoint
    token = st.session_state.datarobot_api_token
    key = st.session_state.datarobot_api_key
    deployment_id = st.session_state.datarobot_deployment_id

    data = {
        "openai_api_key": os.environ["OPENAI_API_KEY"],
        "prompt": prompt,
    }
    if history:
        data["chat_history"] = [
            [entry.get("user", ""), entry.get("chatbot", "")] for entry in history
        ]
    response = make_datarobot_deployment_unstructured_predictions(
        endpoint, data, deployment_id, token, key
    )

    if (
        (not response)
        or ("error" in response)
        or ("completion" in response and "Error" in response["completion"])
    ):
        raise RuntimeError(f"Exception in Custom Model! {response}")
    return response


def update_datarobot_client():
    url = st.session_state.datarobot_prediction_endpoint
    token = st.session_state.datarobot_api_token

    st.experimental_set_query_params(
        endpoint=url,
        token=token,
        api_key=st.session_state.get("datarobot_api_key"),
        deployment_id=st.session_state.get("datarobot_deployment_id"),
    )


def test_datarobot_connection():
    status = st.empty()
    with status:
        st.markdown(":thinking_face: Testing deployment...")
        update_datarobot_client()
        try:
            make_prediction("Testing 123")
        except Exception:
            st.markdown(":exploding_head: Something went wrong!")
            raise
        else:
            st.markdown(":hugging_face: Success!")


def configure_datarobot():
    st.markdown("Configure Your Deployment")
    params = st.experimental_get_query_params()
    defaults = {
        "endpoint": params.get("endpoint", [""])[0],
        "token": params.get("token", [""])[0],
        "api_key": params.get("api_key", [""])[0],
        "deployment_id": params.get("deployment_id", [""])[0],
    }
    st.text_input(
        label="DataRobot Prediction Endpoint",
        value=defaults["endpoint"],
        key="datarobot_prediction_endpoint",
        placeholder="Predictions URL ending in datarobot.com",
        on_change=update_datarobot_client,
    )
    st.text_input(
        label="DataRobot API Token",
        key="datarobot_api_token",
        value=defaults["token"],
        placeholder="Your staging Public API Token",
        type="password",
        on_change=update_datarobot_client,
    )
    st.text_input(
        label="DataRobot API Key",
        key="datarobot_api_key",
        value=defaults["api_key"],
        placeholder="DataRobot-Key for your deployment",
        type="password",
        on_change=update_datarobot_client,
    )
    st.text_input(
        label="Deployment ID",
        key="datarobot_deployment_id",
        value=defaults["deployment_id"],
        placeholder="ID of your deployment",
        on_change=update_datarobot_client,
    )
    if st.button("Test Connection"):
        test_datarobot_connection()

    if st.button("Create shareable link"):
        params = st.experimental_get_query_params()
        params = {key: value[0] for key, value in params.items() if key not in "token"}
        url = urllib.parse.urlunparse(
            (
                "",
                "",
                "",
                None,
                urllib.parse.urlencode(params),
                None,
            )
        )
        st.markdown(f"[Copy this link to share]({url})")
        st.markdown("You will need to share the deployment with your recipients.")


#
# CHAT
#


def clear_chat():
    st.session_state["chat_history"] = []


def run_prompt():
    prompt = st.session_state.latest_prompt
    history = st.session_state.get("chat_history", [])
    # invoke LM
    results = make_prediction(prompt, history=history)
    history.append({"user": prompt, "chatbot": results["completion"]["answer"]})

    st.session_state["chat_history"] = history
    st.session_state["latest_prompt"] = ""


def format_chat_history(chat_history):
    formatted_history = ""
    for entry in chat_history:
        if "user" in entry:
            formatted_history += f"**User:** {entry['user']}\n\n"
        if "chatbot" in entry:
            formatted_history += f"**Chatbot:** {entry['chatbot']}\n\n"
    return formatted_history


def save_chat_response(history, vote: str):
    data = {
        "timestamp": datetime.datetime.now(),
        "vote": vote,
        "chat_history": history,
        "deployment_id": st.session_state.datarobot_deployment_id,
        "endpoint": st.session_state.datarobot_prediction_endpoint,
    }
    record = get_chat_record()
    record.append(data)


def render_saved_chats(ct):
    record = get_chat_record()
    data = record.read()
    if not data.empty:
        data["chat_history"] = data.chat_history.apply(str)
        ct.markdown("## Saved Chat Data")
        ct.markdown(
            "Use the :thumbsup: and :thumbsdown: buttons to mark good and bad conversations. "
            "This data will be used to help understand and build better knowledge "
            "bases and chatbots in the future. \n\n"
            "Below you can see a sample of recent conversations saved by you and your colleagues."
        )
        ct.dataframe(
            data[["deployment_id", "vote", "chat_history"]].iloc[-100:].iloc[::-1]
        )


def render_chat(ct):
    history_area = ct.container()
    voting_area = ct.container()
    prompt_area = ct.container()

    send, _, clear = ct.columns([1, 3, 1])
    do_send = send.button(label="Send")
    do_clear = clear.button(label="Clear Session")

    if do_clear:
        clear_chat()
        history = st.session_state.get("chat_history")

    if do_send:
        run_prompt()

    history = st.session_state.get("chat_history")
    if history:
        thumbsup, thumbsdown, _ = voting_area.columns([1, 1, 15])
        do_upvote = thumbsup.button(label=":thumbsup:")
        do_downvote = thumbsdown.button(label=":thumbsdown:")
        history_area.markdown(format_chat_history(history))

        if do_upvote:
            with voting_area.empty():
                st.markdown(":thinking_face:")
                save_chat_response(history, "upvote")
                st.markdown(":smile: Your response has been recorded!")

        if do_downvote:
            with voting_area.empty():
                st.markdown(":thinking_face:")
                save_chat_response(history, "downvote")
                st.markdown(":disappointed_relieved: We'll try harder next time.")

    prompt_area.text_area(label="Prompt", key="latest_prompt")


def main():
    st.title("DataRobot Custom Model Chat")
    with st.sidebar:
        configure_datarobot()

    chat_area, history_area = st.tabs(["Chat", "Saved Chats"])
    # chat_area = st.container()
    # history_area = st.container()

    render_chat(chat_area)
    render_saved_chats(history_area)


if __name__ == "__main__":
    main()
