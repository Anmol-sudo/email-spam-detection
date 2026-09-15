# Internship Presentation — Speaking Script

**Speaker:** Anmol Sharma · CUID cu25220262 · B.Tech AIML, Section A, Second Year
**Organization:** InternPe · Department of AIML
**Deck:** `Internship_Presentation_Anmol_Sharma.pptx` — 15 slides
**Total time:** ~8–9 minutes (a 5-minute cut is marked at the end)

> Every script below is also embedded in the PPTX as speaker notes, so you can
> read them from Presenter View instead of from paper.

---

## Before you walk in

- [ ] **Replace slide 3** with your actual InternPe certificate image. This is the
      only thing in the deck that is not finished.
- [ ] Open the live app in a browser tab and classify one message, so the server
      is warm if anyone asks for a demo.
- [ ] Memorise five numbers: **98.03 · 99.30 · 87.74 · 3 · 19**
- [ ] If you forget everything else, remember the one line that makes this
      presentation land: *"Only 3 good messages were wrongly flagged — but 19
      spam got through, so my spam recall is 87.7%, not 98%."*

---

## Slide 1 — Title · 20 seconds

> "Good morning. I'm Anmol Sharma, second year B.Tech AIML, Section A, CUID
> cu25220262. Today I'm presenting the work I completed during my internship
> with InternPe, in the AIML department.
>
> My project was an Email and SMS Spam Detection system — built end to end, from
> a raw dataset all the way to a deployed web application that explains its own
> decisions."

*Transition:* "Let me start with what the internship actually involved."

---

## Slide 2 — Introduction · 45 seconds

> "This was a 45-day, project-based internship in Artificial Intelligence and
> Machine Learning. The important word there is project-based — every phase had
> to produce something that actually worked, not just notes or tutorial code.
>
> My project was spam detection: given any SMS or email, classify it as spam, or
> as ham — which is the industry term for a legitimate message.
>
> But I didn't stop at a notebook. The scope covered the entire machine learning
> lifecycle: analysing the data, cleaning the text, comparing six different
> algorithms, tuning them, evaluating properly, making the model explainable,
> writing automated tests, and deploying it.
>
> The final system is 98% accurate, it explains every prediction word by word,
> and it runs as a live web app."

*Transition:* "Here's the certificate from the programme."

---

## Slide 3 — Certificate · 10 seconds

> "This is my internship completion certificate, issued by InternPe."

*Keep this short — don't read the certificate out loud. Move on.*

---

## Slide 4 — Objective · 40 seconds

> "My objectives had six parts.
>
> Build a reliable classifier. Apply the full ML workflow instead of jumping
> straight to a model. Learn to evaluate correctly — because on imbalanced data,
> accuracy actively lies to you, and I'll show you exactly how in a few slides.
>
> Make the model explainable, so it isn't a black box. Actually ship it, so a
> non-technical person can use it. And pick up professional engineering habits
> along the way — version control, pinned dependencies, automated testing,
> documentation."

*Transition:* "First, a little about the organisation."

---

## Slide 5 — About the Company · 35 seconds

> "InternPe is an EdTech company running structured internship and skill
> development programmes across AI and Machine Learning, Data Science and Web
> Development.
>
> I interned in the AIML department, remotely, for 45 days. The format was
> phase-wise — each phase had a task and a deliverable that was reviewed before
> I could move on.
>
> I want to call that out specifically, because it's the reason this project
> ended as working deployed software instead of a folder full of experiments."

*Transition:* "So let's look at the project itself."

---

## Slide 6 — The Project · 45 seconds

> "Spam isn't just annoying — it's the delivery mechanism for phishing and fraud.
>
> But here's the part people miss. Catching spam is easy. I could write ten lines
> of code that flag anything containing the word 'free' and catch most of it.
> **The hard part is catching spam without losing a real message.** If a job offer
> or an invoice ends up in your spam folder, that's a far worse failure than one
> advertisement reaching your inbox.
>
> I broke the work into ten phases, shown here. The dataset is the SMS Spam
> Collection — 5,572 real messages. I trained on 4,457 and held back 1,115, so
> every number I show you later comes from messages the model had never seen."

*Transition:* "The first phase was simply understanding the data — and it found a problem."

---

## Slide 7 — The Data and its biggest problem · 55 seconds ⭐

> "Two findings from my exploratory analysis.
>
> On the left: the dataset is badly imbalanced. 86.6% ham, only 13.4% spam.
>
> Now think about what that means. I can write a model in one line — always
> answer 'ham' — and it scores 86.6% accuracy. And it is completely useless.
>
> *(pause)*
>
> That single statistic shaped every decision that followed: which metrics I
> trusted, how I tuned the model, and an entire phase spent on class imbalance.
>
> On the right: spam genuinely looks different before you even read it. It's
> twice as long. It has fifty times more digits, because of shortcodes and prize
> amounts. Double the uppercase. And 16% contain a link, against almost none in
> ham."

> 💡 **Delivery:** the pause after "completely useless" is the most important
> beat on this slide. Let it land.

*Transition:* "But a model can't read words — so the next step was turning text into numbers."

---

## Slide 8 — Turning Text into Numbers · 55 seconds

> "Six cleaning steps, shown across the top, and you can see the before and after
> underneath.
>
> Three decisions worth explaining.
>
> First, I strip URLs and digits. This sounds destructive, but every spam link is
> a different string, so each one becomes a token the model can never generalise
> from. *Whether* there's a link is signal; *which* link it is, is noise.
>
> Second, I chose lemmatizing over stemming. Stemming gives you non-words like
> 'studi' and 'claimi'. Lemmatizing gives you 'study' and 'claim' — real words.
> That matters because I show these exact tokens to the user later as the
> explanation, and a chart labelled 'studi' would undermine the whole thing.
>
> Third, I included bigrams — adjacent word pairs — so the model learns 'free
> entry' as a single phrase, not just two separate words. You'll see that pay
> off shortly."

*Transition:* "With numbers in hand, I could train something."

---

## Slide 9 — Choosing a Model with Evidence · 55 seconds

> "I didn't want to pick Logistic Regression just because it's the obvious
> choice, so I trained six algorithms on identical features.
>
> The navy bars are overall accuracy. The red bars are spam recall — the number
> that actually matters.
>
> Notice KNN collapses to 39% spam recall. I included it deliberately as a
> control, because distance is nearly meaningless in fifteen thousand sparse
> dimensions, and its score proves it.
>
> Then I tuned with GridSearchCV — 36 configurations, five folds each, 180 model
> fits — scored on F1, not accuracy, because optimising accuracy would have
> tuned the model toward saying 'ham' more often.
>
> Tuning took Logistic Regression from 96.2% to 98.0% accuracy, and its spam
> recall from 73% to 88%.
>
> I chose it over Linear SVM despite a very close score, because it returns true
> probabilities — an SVM gives you a distance from a boundary, not a probability
> — and my confidence display and sensitivity slider both need real
> probabilities. Plus its coefficients are readable, which makes the next slide
> possible."

*Transition:* "So how well does it actually do?"

---

## Slide 10 — Results, and being honest about them · 70 seconds ⭐⭐

> "On 1,115 messages the model had never seen: 98.03% accuracy, ROC-AUC of 99.3.
>
> But I want to break the confusion matrix down, because that headline hides
> something.
>
> Of 960 legitimate messages, only **three** were wrongly flagged as spam. That's
> the error I care most about, and it's very low.
>
> But of 155 real spam messages, **nineteen got through**. So my spam recall is
> **87.7%** — not 98%.
>
> *(pause)*
>
> Why the gap? Accuracy is computed across all 1,115 messages, and 960 of them
> are easy ham. The majority class drowns out the minority. Accuracy flatters
> every model trained on data this imbalanced.
>
> Reporting only the 98% would be hiding the real weak spot. That's the number I
> would work on next if this went into production."

> 💡 **Delivery:** slow right down here. This is the slide that separates you from
> everyone else presenting a 98% number. Volunteering your own weakness before
> anyone asks reads as rigour, and it usually pre-empts the hardest question in
> the room.

*Transition:* "The other thing I insisted on was that the model couldn't be a black box."

---

## Slide 11 — Explainable AI · 50 seconds

> "Logistic regression scores a message as its intercept, plus — for every word —
> that word's TF-IDF value times its coefficient. Which means I can compute
> exactly how much each word contributed. Not an estimate. The actual arithmetic.
>
> Here's what it learned. On the spam side: txt, claim, mobile, prize, win, free,
> urgent, cash. On the ham side: ok, home, road, later, good.
>
> **Nobody told it any of those words.** It rediscovered the entire vocabulary of
> SMS spam from 4,457 training examples.
>
> I also implemented LIME in the notebook — that's a model-agnostic explanation
> technique. But my app doesn't use it, because for a linear model the exact
> answer already exists in closed form. LIME would only approximate it, and it's
> about five hundred times slower. Knowing when *not* to reach for the fancier
> tool was part of the lesson."

*Transition:* "And all of this runs in a live application."

---

## Slide 12 — The Web Application · 50 seconds

> "The final phase was making it usable by someone who doesn't write Python.
>
> This is the live app classifying a real spam message — 93% confidence. And
> underneath, it tells you why: txt, text, win, entry, free. And look at this one
> — **free entry** — as a single feature. That's the bigram I mentioned earlier,
> earning its place.
>
> The app also has an adjustable sensitivity slider, batch mode for scoring a CSV,
> and a model insights tab.
>
> Architecturally I split it in two. `app.py` is purely the interface. `backend.py`
> holds all the inference and imports no Streamlit at all. That means the backend
> is unit-tested without a browser — thirty automated tests — and the same code
> could sit behind a REST API tomorrow."

*Transition:* "So what did I actually learn from all this?"

---

## Slide 13 — Learning · 45 seconds

> "Five things I take away.
>
> One: the model is a small part of the work. Analysis, preprocessing, evaluation
> and deployment took far longer than fitting the classifier — and mattered more.
>
> Two: real NLP feature engineering — TF-IDF, lemmatization, n-grams.
>
> Three, and this is the most valuable: how to evaluate on imbalanced data, where
> accuracy can be high and meaningless at the same time.
>
> Four: explainable AI — and knowing when the simple exact method beats the fancy
> approximate one.
>
> Five: production engineering. Pinning library versions, not depending on the
> working directory, and why thirty automated tests are worth the hour they cost."

*Transition:* "It wasn't smooth, though. Here's what actually cost me time."

---

## Slide 14 — Challenges · 55 seconds

> "Four challenges.
>
> First, the class imbalance made accuracy meaningless, so I changed which
> metrics I trusted and tested three balancing approaches against each other. The
> honest result was that balancing cost more precision than the recall it bought
> — so the shipped model doesn't use SMOTE. I think that's worth more than if it
> had worked, because now it's a measured decision rather than an assumption.
>
> Second, the app crashed on deployment while working perfectly on my machine —
> a path resolved relative to the working directory.
>
> Third, and this one is nastier because it produces no error at all: a saved
> model loaded under a different scikit-learn version can silently change its
> predictions. No crash, no warning. I pinned the exact version and added a
> runtime check.
>
> Fourth, deciding where to draw the line between spam and ham. A probability
> isn't a verdict. I measured that trade-off instead of guessing, and exposed it
> as a slider."

*Transition:* "Let me close."

---

## Slide 15 — Findings, Future Scope & Conclusion · 60 seconds

> "My findings: a simple, well-engineered linear model reached 98% — a neural
> network was never needed here. Accuracy is the wrong headline on imbalanced
> data. Careful preprocessing and bigrams contributed more than switching
> algorithms did. And explainability cost almost nothing on a linear model.
>
> Going forward, I'd fine-tune a transformer like DistilBERT to capture word
> order, retrain on a modern email corpus instead of 2005-era SMS, add character
> n-grams to catch deliberate obfuscation like F-dot-R-dot-E-dot-E, and add
> production monitoring — because spam evolves, and a model that's never
> retrained decays.
>
> In conclusion: over 45 days I built a complete machine learning system rather
> than a single model. But more valuable than the 98% is what it taught me about
> judgement — which metric to trust, when a simpler model is the better
> engineering choice, and why a result you can't explain or deploy isn't actually
> finished.
>
> Thank you to InternPe for a programme built around deliverables. And thank you
> all — I'm happy to take questions."

---

# Likely questions, with answers

Answer in **one sentence first**, then expand only if they look interested.
If you don't know something, say: *"I don't know — my instinct would be X, and
I'd test it by Y."* That scores better than bluffing, every time.

**"Why not use deep learning or BERT?"**
> Because it would be the wrong tool here. These are short messages separated
> almost perfectly by vocabulary, and my linear model already reaches 99.3
> ROC-AUC. BERT would cost a hundred times the compute for maybe a point, and
> I'd lose the exact explanations. I'd reach for it when word order starts to
> matter — distinguishing "not free" from "free".

**"Is 98% accuracy actually good?"**
> Compared to the 86.6% you get by always guessing ham, yes — but accuracy is the
> wrong headline. My spam recall is 87.7%, and that's the number a filter lives
> by. I report both for exactly that reason.

**"Are you overfitting?"**
> I checked rather than assumed. Five-fold cross-validation gave consistent
> scores with a small standard deviation, and the test set was held out from the
> start and only touched at final evaluation.

**"Why Logistic Regression if Linear SVM scored slightly higher?"**
> Probabilities. An SVM gives a distance from the decision boundary, not a
> calibrated probability, and both my confidence display and threshold slider
> need real probabilities. For a fraction of a point of F1, that wasn't a trade
> I wanted.

**"Why didn't SMOTE help?"**
> Because SMOTE on TF-IDF vectors is slightly odd — a synthetic point halfway
> between two spam vectors doesn't correspond to any real sentence. It did what
> it's designed to do: raised recall, lowered precision. With only three false
> positives at baseline, trading precision away wasn't worth it here.

**"What is data leakage, and where could it have happened?"**
> Leakage is when information from the test set influences training. The obvious
> risk here was the vectorizer — if I'd fitted TF-IDF on the whole dataset before
> splitting, the IDF statistics would encode the test messages. I avoided it by
> putting the vectorizer inside a Pipeline, so it's refitted within each fold.

**"Could someone deliberately beat your filter?"**
> Easily, and I'd rather say so. Write "F.R.E.E" or "pr1ze" and the tokens no
> longer match anything in the vocabulary. Character-level n-grams would help;
> real filters also use sender reputation and link analysis, none of which my
> model sees.

**"What was the hardest bug?"**
> The scikit-learn version issue, because it produces no error at all. A model
> saved by one version and loaded by another can silently change its predictions.
> A crash tells you something is wrong; this just quietly gives you different
> answers.

**"What would you do differently?"**
> Set up the train/test split and evaluation harness on day one, before touching
> a model, so every experiment was comparable from the start. And write the tests
> earlier — they found two real bugs, and they'd have found them sooner.

---

# Timing card

| Slide | Topic | Time | Cumulative | Cut for 5 min? |
|---|---|---|---|---|
| 1 | Title | 0:20 | 0:20 | keep |
| 2 | Introduction | 0:45 | 1:05 | keep |
| 3 | Certificate | 0:10 | 1:15 | keep |
| 4 | Objective | 0:40 | 1:55 | trim to 0:20 |
| 5 | About the Company | 0:35 | 2:30 | trim to 0:20 |
| 6 | The Project | 0:45 | 3:15 | keep |
| 7 | **The Data** | 0:55 | 4:10 | keep — never cut |
| 8 | Preprocessing | 0:55 | 5:05 | trim to 0:30 |
| 9 | Model Comparison | 0:55 | 6:00 | trim to 0:30 |
| 10 | **Results** | 1:10 | 7:10 | keep — never cut |
| 11 | Explainable AI | 0:50 | 8:00 | trim to 0:30 |
| 12 | Web Application | 0:50 | 8:50 | keep |
| 13 | Learning | 0:45 | 9:35 | trim to 0:25 |
| 14 | Challenges | 0:55 | 10:30 | trim to 0:30 |
| 15 | Conclusion | 1:00 | 11:30 | keep |

**Full version:** ~11 minutes if you speak every word. Realistically 8–9, because
you'll naturally compress.

**If you only get 5 minutes:** slides 1, 2, 6, 7, 10, 12, 15. The data problem,
the honest results, and the working app are the whole story — everything else is
supporting evidence.

**If you only get 60 seconds:** slide 10 alone. "I built a spam classifier that's
98% accurate. But only 3 of 960 good messages were wrongly flagged, while 19 of
155 spam got through — so the number that actually matters, spam recall, is 87.7%,
not 98. Knowing the difference is what I learned."
