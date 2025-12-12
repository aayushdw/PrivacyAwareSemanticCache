import pandas as pd
from typing import List, Dict, Tuple


class UnionFind:
    """Union-Find data structure for grouping semantically similar questions."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1


def get_unique_questions(csv_path: str) -> List[Dict[str, any]]:
    """
    Extracts semantically unique questions from the QQP dataset.

    Args:
        csv_path: Path to the questions.csv file

    Returns:
        List of dictionaries containing unique questions with their IDs
    """
    df = pd.read_csv(csv_path)

    # Build a union-find structure to group duplicate questions
    uf = UnionFind()

    # Map question IDs to their text
    qid_to_question = {}

    # Process each row
    for _, row in df.iterrows():
        qid1 = row['qid1']
        qid2 = row['qid2']
        question1 = row['question1']
        question2 = row['question2']
        is_duplicate = row['is_duplicate']

        # Store question texts
        if qid1 not in qid_to_question:
            qid_to_question[qid1] = question1
        if qid2 not in qid_to_question:
            qid_to_question[qid2] = question2

        # Union duplicate questions
        if is_duplicate == 1:
            uf.union(qid1, qid2)

    # Group questions by their root representative
    groups = {}
    for qid in qid_to_question.keys():
        root = uf.find(qid)
        if root not in groups:
            groups[root] = []
        groups[root].append(qid)

    # Select one representative question from each group
    unique_questions = []
    for root, qids in groups.items():
        # Use the root as the representative
        representative_qid = root
        unique_questions.append({
            'qid': representative_qid,
            'question': qid_to_question[representative_qid],
            'group_size': len(qids)
        })

    return unique_questions


def save_unique_questions(csv_path: str, output_path: str) -> Tuple[int, int]:
    """
    Extracts unique questions and saves them to a file.
    """
    unique_questions = get_unique_questions(csv_path)

    # Convert to DataFrame and save
    df = pd.DataFrame(unique_questions)
    df.to_csv(output_path, index=False)

    # Calculate statistics
    total_questions = df['group_size'].sum()
    unique_count = len(unique_questions)

    print(f"Total questions: {total_questions}")
    print(f"Unique questions: {unique_count}")
    print(f"Reduction: {100 * (1 - unique_count/total_questions):.2f}%")


if __name__ == "__main__":
    # Example usage
    csv_path = "questions.csv"
    output_path = "unique_questions.csv"
    save_unique_questions(csv_path, output_path)
