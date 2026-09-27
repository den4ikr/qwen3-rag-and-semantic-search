import sys

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import chromadb

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QTextEdit,
    QLineEdit,
    QPushButton,
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

MODEL_NAME = 'Qwen/Qwen2.5-1.5B-Instruct'

embedding_model = SentenceTransformer(
    'Qwen/Qwen3-Embedding-0.6B'
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# --------------------------------------------------
# DATABASE
# --------------------------------------------------
db_client = chromadb.PersistentClient('./resources')
resources_collection = db_client.get_collection('resources')

# --------------------------------------------------
# CHAT
# --------------------------------------------------
messages = [
    {
        'role': 'system',
        'content': 'You are my JARVIS from iron man movie'
    }
]

def get_context_from_db(user_input):
    user_embedding = embedding_model.encode(user_input)

    res = resources_collection.query(
        query_embeddings=[user_embedding],
        n_results=1,
        include=['documents', 'metadatas']
    )

    return (
        res['documents'][0][0],
        res['metadatas'][0]
    )


def generate_prompt(user_input):
    documents, metadatas = get_context_from_db(user_input)
    prompt_start = (
        'Answer the question based on the context below.\n\n'
        'Context:\n'
    )
    prompt_end = (
        f'\n\nQuestion: {user_input}\n'
        f'Answer:'        
    )
    prompt = prompt_start + documents + prompt_end

    return prompt


def generate_answer(user_input):
    prompt = generate_prompt(user_input)
    _, metadatas = get_context_from_db(user_input)
    messages.append({
        'role': 'user',
        'content': prompt
    })

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    tokens = tokenizer(
        chat_text,
        return_tensors='pt'
    )

    res = model.generate(
        **tokens,
        max_new_tokens=300
    )

    input_length = tokens['input_ids'].shape[1]

    model_answer = res[0][input_length:]

    answer = tokenizer.decode(
        model_answer,
        skip_special_tokens=True
    )

    return f"""
        {answer}\n
        Tags: {metadatas}
    """


# --------------------------------------------------
# WORKER THREAD
# --------------------------------------------------

class Worker(QThread):
    finished = pyqtSignal(str)

    def __init__(self, user_input):
        super().__init__()

        self.user_input = user_input

    def run(self):
        answer = generate_answer(self.user_input)

        self.finished.emit(answer)


# --------------------------------------------------
# UI
# --------------------------------------------------

app = QApplication(sys.argv)

window = QWidget()
window.setWindowTitle("JARVIS")
window.resize(500, 600)

layout = QVBoxLayout()

chat = QTextEdit()
chat.setReadOnly(True)

user_input = QLineEdit()
user_input.setPlaceholderText("Ask JARVIS...")

send_button = QPushButton("Send")

layout.addWidget(chat)
layout.addWidget(user_input)
layout.addWidget(send_button)

window.setLayout(layout)


# --------------------------------------------------
# SEND MESSAGE
# --------------------------------------------------

def send_message():
    text = user_input.text().strip()
    if not text:
        return

    chat.append(f'YOU: {text}')
    user_input.clear()
    chat.append('JARVIS: Thinking...')

    send_button.setEnabled(False)
    user_input.setEnabled(False)

    worker = Worker(text)

    def show_answer(answer):
        cursor = chat.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)

        chat.setTextCursor(cursor)
        chat.append(f'JARVIS: {answer}')

        send_button.setEnabled(True)
        user_input.setEnabled(True)
        user_input.setFocus()

    worker.finished.connect(show_answer)
    window.worker = worker
    worker.start()


send_button.clicked.connect(send_message)

# --------------------------------------------------
# START UI
# --------------------------------------------------
window.show()
app.exec()