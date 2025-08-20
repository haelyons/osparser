import json
import sys
import anthropic
from typing import Dict

def get_api_key() -> str:
    """Read the Anthropic API key from .keys file"""
    try:
        with open('.keys', 'r') as f:
            for line in f:
                if line.startswith('ANTHROPIC_API_KEY='):
                    return line.split('=', 1)[1].strip()
        raise ValueError("ANTHROPIC_API_KEY not found")
    except FileNotFoundError:
        raise FileNotFoundError(".keys file not found")

def call_claude_relevance_judge(question: str, summary: str, api_key: str) -> Dict:
    """Judge document relevance based on summary"""
    prompt = f"""You are an impartial AI document relevance judge. Your task is to evaluate how relevant the SOURCE DOCUMENT is to a user's question, based on the summary that was generated from that document.

**User Question:**
"{question}"

**Summary Generated from Source Document:**
"{summary}"

**Instructions:**
1. **Focus on Document Content, Not Summary Quality:** You are NOT judging how well-written the summary is. Instead, judge how much relevant information the original document contains about the user's question, as revealed by the summary.

2. **Rationale First:** Provide a brief rationale explaining how much relevant content the source document appears to contain about the user's question, based on what the summary reveals.

3. **Assign Score:** Provide a numerical score on a scale of 1 to 5 based on the following criteria:
   - 1: **Document Contains No Relevant Information.** The summary indicates the document does not address the question topic at all, or only mentions it in passing without substance.
   - 2: **Document Contains Minimal Relevant Information.** The summary reveals only brief mentions or tangential references to the question topic.
   - 3: **Document Contains Some Relevant Information.** The summary shows the document partially addresses the question with some useful content, but is incomplete.
   - 4: **Document Contains Substantial Relevant Information.** The summary reveals the document provides comprehensive information directly addressing the question.
   - 5: **Document is Highly Relevant.** The summary shows the document is specifically focused on the question topic with detailed, comprehensive coverage.

**Important:** A summary that clearly states "the document does not contain information about X" should receive a LOW score (1-2), because this indicates the SOURCE DOCUMENT lacks relevant content, even though the summary itself is accurate and well-written.

Your final output **MUST** be a valid JSON object with two keys: "rationale" (a string) and "score" (an integer from 1 to 5).

Example format:
{{
  "rationale": "Based on the summary, the source document contains extensive information about climate change impacts, including specific data on temperature rises and ecosystem effects...",
  "score": 4
}}"""

    try:
        response = anthropic.Anthropic(api_key=api_key).messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = json.loads(response.content[0].text.strip())
        result['normalized_score'] = (result['score'] - 1) / 4.0
        return result
        
    except Exception as e:
        return {"rationale": f"ERROR: {e}", "score": 1, "normalized_score": 0.0}

def main():
    """Main function to run the relevance judge."""
    if len(sys.argv) != 3:
        print("Usage: python judge.py <question> <summary>")
        sys.exit(1)
    
    question, summary = sys.argv[1], sys.argv[2]
    
    try:
        api_key = get_api_key()
        result = call_claude_relevance_judge(question, summary, api_key)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
