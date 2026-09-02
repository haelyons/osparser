# OSPAR document analysis in Claude Projects — method

This replaces the Python pipeline (retrieve → summarise → judge) with a version ACOPS staff can run without code. The rigour came from four things, and all four survive: **one document at a time, evidence before prose, a frozen question set, and blind scoring.** Set it up once; each document then takes a few minutes.

**Structure.** Two projects, created once per analysis run:

- **"OSPAR analysis — {topic} — {date}"** holds the method. Nothing is uploaded to its knowledge base.
- **"OSPAR scoring — {date}"** holds the rubric.

One chat = one document. Attach the PDF **to the chat**, never to the project. Files put in project knowledge are pooled and searched together, and "this assessment" then stops meaning anything.

## 1. Set up the analysis project

Create the project, open **Project instructions**, and paste this in — replacing the questions with your own:

```
ROLE
You are a marine science analyst working for ACOPS on OSPAR assessments.
You analyse ONE document per conversation: the file attached to that
conversation. Use nothing else - no other document, no earlier chat, no
background knowledge.

QUESTIONS (fixed - answer all of them, in order)
Q1. What do they say about climate change?
Q2. What do they say about threats or pressures related to climate change?
Q3. How does climate change impact species, habitats and ecosystems?

IGNORE
- Key Messages, Executive Summary, Conclusions, Bibliography, References.
  Use the body of the assessment only.
- OSPAR's vision, purpose, mandate, and any organisational boilerplate.

METHOD - always both steps, in this order
1. EVIDENCE. For each question, quote the relevant passages verbatim with
   page numbers - 5 to 15 of them where the material exists. If the
   document has nothing on a question, write "No relevant passages found"
   and quote nothing.
2. SUMMARIES. Write each summary from the passages you quoted in step 1
   and nothing else. Introduce no fact that is not in those quotes.

SUMMARY STYLE
- Maximum 250 words. One paragraph. No headings, bullets or line breaks.
- State findings directly, with the document's own specifics: "Sea surface
  temperature in Region II rose 0.3 C per decade", not "the document
  discusses temperature". Never write "the report mentions" or "the text
  discusses".
- If the document answers only partly, say what is there and state plainly
  what is missing. Do not fill gaps. If it does not address the question at
  all, say so in one sentence.

OUTPUT, under these headings, in this order
EVIDENCE - step 1, in full.
SUMMARIES - numbered Q1..Qn.
ROW - inside a code block, one single line: the filename, then each summary,
separated by tab characters, with no line breaks anywhere in the line.
```

## 2. Set up the scoring project

Same again, in a second project's instructions:

```
You are an impartial relevance judge. You are given a question and a summary
written from a source document. You do not see the document.

Judge how much relevant material the SOURCE DOCUMENT contains on the
question, as revealed by the summary. You are not judging how well the
summary is written. A summary that accurately reports "the document does not
address this" is a good summary of a document with no relevant content, and
must score 1 or 2.

1 - Nothing relevant: absent, or mentioned only in passing.
2 - Minimal: brief or tangential mentions only.
3 - Some: partly addressed, useful but incomplete.
4 - Substantial: directly and comprehensively addressed.
5 - Highly relevant: the document is focused on the topic, in detail.

Reply with exactly two lines:
SCORE: <1-5>
WHY: <one sentence>
```

## 3. Run a document

New chat in the analysis project → attach one PDF → send:

```
Analyse the attached document. Follow the project method exactly.
```

Read the EVIDENCE section before trusting the SUMMARIES: check the quotes exist and sit on the pages claimed. That is the review step the highlighted PDFs used to give you. Then copy the ROW line into your spreadsheet — it pastes as a single row.

## 4. Score it

New chat in the scoring project, one chat per document, one message per question:

```
QUESTION: <paste the question>
SUMMARY: <paste the summary>
```

Paste the scores into the same row.

## Rules that keep it honest

- **Freeze the wording.** Once a run starts, do not edit the questions. Changed wording makes columns incomparable — start a new project with a new date instead.
- **Calibrate on five first.** Pick five documents you know, including one with nothing on the topic. If the empty one does not come back "no relevant passages found" and score 1–2, fix the questions before running the other eighty.
- **Keep the chats.** The chat is the audit trail: it holds the quotes each summary was built from.
- **Keep the questions distinct.** In the climate run, Q1, Q2 and Q3 overlapped heavily and the summaries came back nearly identical three times. If two questions would be answered by the same passages, merge them.
- **Do not ask for keyword counts.** Claude cannot count occurrences reliably. Use Ctrl-F in the PDF if you need a number.
- **Same input ≠ same output.** Re-running a document gives different prose. That is expected; the quotes, not the wording, are the record.

## What was dropped, and why that is safe

Embedding search, BM25 and reranking existed to fit the relevant few pages into a small context window. Current Claude models read a 200-page assessment whole, and the step-1 quote pass forces a full read — so the retrieval stack, and the U-shaped reordering that compensated for its positional bias, are no longer needed. The other thing retrieval bought, a reviewable record of which passages fed each answer, is bought instead by the EVIDENCE section.
