from transformers import AutoTokenizer, AutoModel 
import torch

# ---- Detector Class ----
class UniXcoderApproachDetector:
    def __init__(self, reference_solutions):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
        self.model = AutoModel.from_pretrained("microsoft/unixcoder-base")
        
        self.reference_embeddings = {}
        for approach, solutions in reference_solutions.items():
            embeddings = []
            for solution in solutions:
                embedding = self.get_embedding(solution)
                embeddings.append(embedding)
            self.reference_embeddings[approach] = torch.mean(torch.stack(embeddings), dim=0)
    
    def get_embedding(self, code):
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.pooler_output.squeeze()
        return embedding
    
    def detect_approach(self, user_code):
        user_embedding = self.get_embedding(user_code)
        similarities = {}
        for approach, ref_embedding in self.reference_embeddings.items():
            similarity = torch.cosine_similarity(
                user_embedding.unsqueeze(0),
                ref_embedding.unsqueeze(0)
            ).item()
            similarities[approach] = similarity
        return similarities

# ---- Reference Solutions for LIS ----
brute_force = """
def lis_recursive(nums):
    def helper(i, prev):
        if i == len(nums):
            return 0
        taken = 0
        if nums[i] > prev:
            taken = 1 + helper(i + 1, nums[i])
        not_taken = helper(i + 1, prev)
        return max(taken, not_taken)
    return helper(0, float('-inf'))
"""

top_down_dp = """
def lis_top_down(nums):
    memo = {}
    def helper(i, prev):
        if i == len(nums):
            return 0
        if (i, prev) in memo:
            return memo[(i, prev)]
        taken = 0
        if nums[i] > prev:
            taken = 1 + helper(i + 1, nums[i])
        not_taken = helper(i + 1, prev)
        memo[(i, prev)] = max(taken, not_taken)
        return memo[(i, prev)]
    return helper(0, float('-inf'))
"""

bottom_up_dp = """
def lis_bottom_up(nums):
    n = len(nums)
    dp = [1] * n
    for i in range(n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
"""

binary_search_dp = """
from bisect import bisect_left
def lis_binary_search(nums):
    sub = []
    for x in nums:
        i = bisect_left(sub, x)
        if i == len(sub):
            sub.append(x)
        else:
            sub[i] = x
    return len(sub)
"""

reference_solutions = {
    "Brute Force": [brute_force],
    "Top-Down DP": [top_down_dp],
    "Bottom-Up DP": [bottom_up_dp],
    "Binary Search DP": [binary_search_dp]
}

# ---- Instantiate Detector ----
detector = UniXcoderApproachDetector(reference_solutions)

# ---- Candidate Codes ----
candidate_full = """
def lis_candidate(nums):
    n = len(nums)
    dp = [1] * n
    for i in range(n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
"""

candidate_partial = """
dp = [1] * len(nums)
for i in range(len(nums)):
    for j in range(i):
        if nums[i] > nums[j]:
            dp[i] = max(dp[i], dp[j] + 1)
"""

candidate_hybrid = """
def lis_candidate(nums):
    memo = {}
    def helper(i):
        if i in memo:
            return memo[i]
        best = 1
        for j in range(i):
            if nums[i] > nums[j]:
                best = max(best, helper(j) + 1)
        memo[i] = best
        return best
    dp = [helper(i) for i in range(len(nums))]
    return max(dp)
"""

# ---- Run Detection ----
print("=== Full Candidate (Bottom-Up DP) ===")
print(detector.detect_approach(candidate_full), "\n")

print("=== Partial Candidate (Incomplete Bottom-Up DP) ===")
print(detector.detect_approach(candidate_partial), "\n")

print("=== Hybrid Candidate (Top-Down + Bottom-Up DP) ===")
print(detector.detect_approach(candidate_hybrid), "\n")
