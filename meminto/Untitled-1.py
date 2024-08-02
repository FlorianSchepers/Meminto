def chunk_list(integers, soft_limit, hard_limit):
    chunks = []
    current_chunk = []
    current_sum = 0

    for num in integers:
        if current_sum + num > hard_limit:
            if current_sum >= 0.9 * soft_limit:
                chunks.append(current_chunk)
                current_chunk = [num]
                current_sum = num
            else:
                # Try to fit the number in the next chunk
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [num]
                current_sum = num
        else:
            current_chunk.append(num)
            current_sum += num

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# Example usage
integers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 1, 1, 1, 1, 1]
soft_limit = 10
hard_limit = 15

result = chunk_list(integers, soft_limit, hard_limit)
print(result)
