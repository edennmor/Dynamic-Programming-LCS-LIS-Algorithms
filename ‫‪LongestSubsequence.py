import numpy as np



def number_of_lcs(A: list[int], B: list[int])->int:
    """
    Finds all longest common subsequences (LCS) by traversing the DP table from the end (bottom-right),
    exploring all valid paths, and including cases where we move left instead of up when values are equal.

    param A: List of integers (first sequence)
    param B: List of integers (second sequence)
    return: A number of all longest common subsequences (LCS)
    """
    n, m = len(A), len(B)
    dp = np.zeros((n + 1, m + 1), dtype=int)

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if A[i - 1] == B[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # List to store all LCS sequences
    all_lcs_sequences = []
    lcs_indices = []

    index_tot = dp[n][m]
    # Explore alternative paths by moving left when dp values are equal
    shift = 0
    while shift < min(n, m) and dp[n][m - shift] == dp[n][m]:
        i, j = n , m - shift

        lcs_algo2 = [""] * (index_tot + 1)
        lcs_algo2[index_tot] = ""
        temp_indices = set()

        index=index_tot
        while i > 0 and j > 0:
            if A[i - 1] == B[j - 1]:
                lcs_algo2[index - 1] = A[i - 1]
                temp_indices.add((i,j))
                i -= 1
                j -= 1
                index -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1


        # Check if at least one index pair is different
        is_different = False
        for idx in temp_indices:
            if idx not in lcs_indices:
                is_different = True
                break

        # Add only if indices are different
        if is_different:
            all_lcs_sequences.append([elem for elem in lcs_algo2 if elem])
            lcs_indices.extend(temp_indices)


        shift += 1

    return len(all_lcs_sequences)




def all_lcs(A: list[int], B: list[int], teta: int)->list[list[int]]:
    """
    Finds all longest common subsequences (LCS) by traversing the DP table from the end (bottom-right),
    exploring all valid paths, and including cases where we move left instead of up when values are equal.

    param A: List of integers (first sequence)
    param B: List of integers (second sequence)
    return: List of all longest common subsequences (LCS)
    """
    n, m = len(A), len(B)
    dp = np.zeros((n + 1, m + 1), dtype=int)

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if A[i - 1] == B[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # List to store all LCS sequences
    all_lcs_sequences = []
    lcs_indices = []

    index_tot = dp[n][m]
    # Explore alternative paths by moving left when dp values are equal
    shift = 0
    while shift < teta and dp[n][m - shift] == dp[n][m]:
        i, j = n , m - shift

        lcs_algo2 = [""] * (index_tot + 1)
        lcs_algo2[index_tot] = ""
        temp_indices = set()

        index=index_tot
        while i > 0 and j > 0:
            if A[i - 1] == B[j - 1]:
                lcs_algo2[index - 1] = A[i - 1]
                temp_indices.add((i,j))
                i -= 1
                j -= 1
                index -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1


        # Check if at least one index pair is different
        is_different = False
        for idx in temp_indices:
            if idx not in lcs_indices:
                is_different = True
                break

        # Add only if indices are different
        if is_different:
            all_lcs_sequences.append([elem for elem in lcs_algo2 if elem])
            lcs_indices.extend(temp_indices)


        shift += 1

    return all_lcs_sequences





def all_unique(A: list[int], B: list[int], teta: int)->list[list[int]]:
    """
    Finds all longest common subsequences (LCS) by traversing the DP table from the end (bottom-right),
    exploring all valid paths, and including cases where we move left instead of up when values are equal.

    param A: List of integers (first sequence)
    param B: List of integers (second sequence)
    return: List of unique longest common subsequences (LCS)
    """
    n, m = len(A), len(B)
    dp = np.zeros((n + 1, m + 1), dtype=int)

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if A[i - 1] == B[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # List to store all LCS sequences
    all_lcs_sequences = []
    lcs_indices = []

    index_tot = dp[n][m]
    # Explore alternative paths by moving left when dp values are equal
    shift = 0
    while shift < teta and dp[n][m - shift] == dp[n][m]:
        i, j = n , m - shift

        lcs_algo2 = [""] * (index_tot + 1)
        lcs_algo2[index_tot] = ""
        temp_indices = set()

        index=index_tot
        while i > 0 and j > 0:
            if A[i - 1] == B[j - 1]:
                lcs_algo2[index - 1] = A[i - 1]
                temp_indices.add((i,j))
                i -= 1
                j -= 1
                index -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1


        # Check if at least one index pair is different
        is_different = False
        for idx in temp_indices:
            if idx not in lcs_indices:
                is_different = True
                break

        # Add only if indices are different and if not contained in all_lcs_sequences
        if is_different and [elem for elem in lcs_algo2 if elem] not in all_lcs_sequences:
            all_lcs_sequences.append([elem for elem in lcs_algo2 if elem])
            lcs_indices.extend(temp_indices)


        shift += 1

    return all_lcs_sequences



def lenght_of_lis(A: list[int], C: list[int])->int:

    """
    Finds the length of the longest strictly increasing subsequence.

    param A: List of integers (first sequence)
    param C: List of binary integers (second sequence)

    return: Length of the longest increasing subsequence
    """

    # Step 1: Filtering elements of A  where C[i] == 1
    new_A = [A[i] for i in range(len(A)) if C[i] == 1]
    n = len(new_A)

    if n == 0:
        return []

    dp = [1] * n  # Initialization of dp array by ones
    # Filling dp array
    for i in range(1, n):
        for j in range(i):
            if new_A[i] > new_A[j]:
                dp[i] = max(dp[i], dp[j] + 1)

    # Return the maximum value in the dp array
    return max(dp)



def number_of_lis(A: list[int], C: list[int])->int:

    """
    Finds the number of the longest strictly increasing subsequences.

    param A: List of integers (first sequence)
    param C: List of binary integers (second sequence)

    return:  The number of distinct longest increasing subsequences
    """

    # Step 1: Filtering elements of A where C[i] == 1
    new_A = [A[i] for i in range(len(A)) if C[i] == 1]
    n = len(new_A)

    if n == 0:
        return []

    # length[i] = length of LIS ending at index i
    len_1 = [1] * n

    # counter[i] = how many LIS ending at index i with length of len_1[i]
    counter = [1] * n

    # Calculating the length and counting arrays
    for i in range(n):
        for j in range(i):
            if new_A[i] > new_A[j]:
                if len_1[j] + 1 > len_1[i]:# If a longer subsequence is found
                    len_1[i] = len_1[j] + 1
                    counter[i] = counter[j]
                elif len_1[j] + 1 == len_1[i]: # If another subsequence with the same length is found
                    counter[i] += counter[j]

    # Finding the maximum length
    max_length = max(len_1)

    # total count of  all LIS with the maximum length
    total_count = sum(counter[i] for i in range(n) if len_1[i] == max_length)

    return total_count

def all_lis(A: list[int], C: list[int], teta: int) -> list[list[int]]:
    """
    Find all longest increasing subsequence paths up to teta limit,
    including paths with the same values but different indices.

    Args:
        param A: List of integers (first sequence)
        param C: List of binary integers
        param teta: Maximum number of LIS paths to return

        return: List of all LIS paths, limited to teta paths
    """
    # Step 1: Filter elements from A based on C
    new_A = [A[i] for i in range(len(A)) if C[i] == 1]
    n = len(new_A)

    if n == 0 or teta <= 0:
        return []

    # Step 2: Compute LIS length using DP
    dp = [1] * n  # dp[i] stores length of LIS ending at index i

    for i in range(1, n):
        for j in range(i):
            if new_A[i] > new_A[j]:
                dp[i] = max(dp[i], dp[j] + 1)

    # Step 3: Find maximum LIS length
    max_lis_length = max(dp)

    # Step 4: Construct adjacency graph for LIS paths
    graph = [[] for _ in range(n + 1)]

    for i in range(n):
        for j in range(i + 1, n):
            if new_A[j] > new_A[i] and dp[j] == dp[i] + 1:
                graph[i].append(j)

    start_indices = [i for i in range(n) if dp[i] == 1]

    # Create a virtual start node pointing to all LIS starting indices
    for idx in start_indices:
        graph[n].append(idx)

    # Step 5: Use DFS to retrieve LIS sequences up to teta limit
    all_paths = []

    def dfs(node, path, path_indices, target_length):
        if len(all_paths) >= teta:
            return

        if len(path) == target_length:
            all_paths.append((path[:], path_indices[:]))
            return

        for next_node in graph[node]:
            path.append(new_A[next_node])
            path_indices.append(next_node)
            dfs(next_node, path, path_indices, target_length)
            path.pop()
            path_indices.pop()

    dfs(n, [], [], max_lis_length)

    return [path for path, _ in all_paths]

def all_unique_lis(A: list[int], C: list[int], teta: int) -> list[list[int]]:
    """
    Finds the number of the longest strictly increasing subsequences.

    param A: List of integers (first sequence)
    param C: List of binary integers (second sequence)
    param teta: Maximum number of LIS paths to return

    return: List of LIS paths, limited to teta paths
    """

    # Extracting relevant elements from A based on C
    new_A = [A[i] for i in range(len(A)) if C[i] == 1]
    size = len(new_A)

    if size == 0 or teta <= 0:
        return []

    # Computing LIS length for each position using DP
    lis_length = [1] * size  # Stores LIS length at each index

    for i in range(1, size):
        for j in range(i):
            if new_A[i] > new_A[j]:
                lis_length[i] = max(lis_length[i], lis_length[j] + 1)

    # Identifying the maximum LIS length
    max_lis = max(lis_length)

    # Constructing a graph to track LIS paths
    adjacency_list = [[] for _ in range(size + 1)]

    for i in range(size):
        for j in range(i + 1, size):
            if new_A[j] > new_A[i] and lis_length[j] == lis_length[i] + 1:
                adjacency_list[i].append(j)

    start_nodes = [i for i in range(size) if lis_length[i] == 1]

    # Virtual starting node connects to all valid starting indices
    for index in start_nodes:
        adjacency_list[size].append(index)

    # Using DFS to retrieve all LIS sequences up to teta limit
    results = []
    seen_sequences = set()

    def explore(node, path, target_length):
        if len(results) >= teta:
            return

        if len(path) == target_length:
            sequence_tuple = tuple(path)
            if sequence_tuple not in seen_sequences:
                seen_sequences.add(sequence_tuple)
                results.append(path[:])
            return

        for neighbor in sorted(adjacency_list[node], key=lambda x: new_A[x]):
            if len(results) >= teta:
                return

            path.append(new_A[neighbor])
            explore(neighbor, path, target_length)
            path.pop()

    explore(size, [], max_lis)

    return results


if __name__ == "__main__":
    # Example usage


    A = [3, 4, 2, 1, 3, 5]
    B = [6,-3,8]

    A = [5, 7, 9, 1, 5]
    B = [5, 7, 1, 9, 8]

    A = [3,4,2,1,3,5]
    B = [1,2,4,2,2]
    # Run the function
    lcs_results = number_of_lcs(A, B)
    print("Number of LCS sequences:", lcs_results)

    # Run the function
    lcs1_results = all_lcs(A, B, 3)
    print("All Longest Common Subsequences:", lcs1_results)

    # Run the function
    lcs2_results = all_unique(A, B, 3)
    print("All unique Common Subsequences:", lcs2_results)

    A = [-5, 4, 5, 3, 6, 7, 9, 61]
    C = [1, 1, 1, 1, 0, 0, 1, 1]

    A = [3, 10, 5, 11, 7, 5, 7, 100]
    C = [0, 1, 1, 1, 1, 1, 1, 0]

    result = lenght_of_lis(A, C)
    print("LIS length:", result)

    result2 = number_of_lis(A, C)
    print("The total count of all LIS with the maximum length:", result2)

    teta = 8
    result3 = all_lis(A, C, teta)
    print(f"Found all {len(result3)} LIS paths (limited to {teta}):")
    for path in result3:
        print(path)
    print()

    result = all_unique_lis(A, C, teta)
    print(f"Found unique {len(result)} LIS paths:")
    for path in result:
        print(path)
    print()

