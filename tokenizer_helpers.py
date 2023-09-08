import tiktoken

def get_token_count(content):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(content))

def get_token_count_from_json(json_data):
    messages = json_data["messages"]
    enc = tiktoken.get_encoding("cl100k_base")
    token_sizes_per_message = [len(enc.encode(message["content"])) for message in messages]
    return sum(token_sizes_per_message)
