def lcs_string(s1: str, s2: str) -> str:
    """
    finds the longest common subsequence (LCS) of two strings.

    the LCS is a sequence that appears in both strings, in the same relative order,
    but not necessarily contiguous. this effectively finds the "common string
    by deleting differences".

    args:
        s1: the first string.
        s2: the second string.

    returns:
        the longest common subsequence string.
    """
    m = len(s1)
    n = len(s2)

    # create a 2d array (dp table) to store lengths of LCS for subproblems
    # dp[i][j] will store the length of LCS of s1[0...i-1] and s2[0...j-1]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # fill the dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                # if characters match, extend the LCS
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                # if characters don't match, take the maximum LCS from
                # considering s1 without its last char, or s2 without its last char
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # backtrack to reconstruct the LCS string
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            # if characters match, it's part of LCS, add it and move diagonally
            lcs.append(s1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            # if char s1[i-1] is not part of LCS, move up
            i -= 1
        else:
            # if char s2[j-1] is not part of LCS, move left
            j -= 1

    # the LCS was constructed in reverse, so reverse it back
    return "".join(lcs[::-1])