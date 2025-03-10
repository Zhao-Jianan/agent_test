from langserve import RemoteRunnable

client = RemoteRunnable("http://localhost:8000/chainDemo")
print(client.invoke({"language": "English", "text": "我们一起去逛街"}))
