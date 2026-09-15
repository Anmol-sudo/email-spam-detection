# Presentation Script — Spam Mail Prediction using Machine Learning

**Speaker:** Arjun
**Length:** 7 minutes for the deck + 2 minutes live demo (a 5-minute cut is marked below)
**Deck:** 13 slides, generated from `gamma_ai_ppt_prompt.txt`

---

## Before you start

- [ ] Open the deployed Streamlit app in a tab **and classify one message** — this
      wakes the container and downloads the NLTK data so the live demo is instant
- [ ] Have the notebook open in a second tab, scrolled to the confusion matrix,
      in case someone asks to see the code
- [ ] Know these five numbers cold: **98.03 · 99.30 · 87.74 · 3 · 19**
- [ ] Decide now: if the demo fails, you go to the screenshot on slide 11 and keep
      talking. Never debug live.

**Pacing:** roughly 30 seconds per slide, 60–75 on slides 9 and 11. If you are
running long, cut slide 3 and slide 12 — the content survives without them.

---

## Slide 1 — Title · 20 seconds

> "Good morning. I'm Arjun, and over the last 45 days I built a spam detection
> system — end to end, from a raw CSV of text messages to a deployed web app
> that not only classifies a message, but tells you *why* it made that call.
>
> I want to show you three things today: how the model works, how well it
> actually performs — including where it falls short — and then I'll classify a
> live message in front of you."

*Transition: "So let's start with why this problem is harder than it looks."*

---

## Slide 2 — The problem · 40 seconds

> "Everyone in this room deletes spam every day. Roughly half of all email
> traffic worldwide is spam, and it's not just clutter — phishing messages are
> one of the most common ways credentials get stolen.
>
> But here's the part people miss. Catching spam is easy. I could write ten
> lines of code that flag anything containing the word 'free' and catch most of
> it. **The hard part is catching spam without losing a real message.** If a job
> offer or an invoice lands in your spam folder, that's a far worse failure than
> one advert reaching your inbox.
>
> So the goal wasn't just accuracy. It was a system that classifies any message,
> explains its reasoning, and lets a user decide how aggressive it should be."

*Transition: "Here's how I got there."*

---

## Slide 3 — Roadmap · 25 seconds *(cut this slide in the 5-minute version)*

> "I broke it into ten phases across three stages. First, understand the data.
> Then build — text preprocessing, comparing six models, evaluating properly,
> tuning, and handling imbalance. Then ship — saving the model, building the
> app, making it explainable, and documenting and testing it.
>
> The key point is that each phase fed the next. What I found in the EDA is what
> forced the decisions in phases four and six."

*Transition: "So — what did the data actually say?"*

---

## Slide 4 — The data · 40 seconds

> "I used the SMS Spam Collection dataset: 5,572 real text messages, each
> labelled spam or ham — 'ham' being the industry term for a legitimate message.
>
> And immediately there's a problem. **86.6% of these messages are ham. Only
> 13.4% are spam.**
>
> Think about what that means. I can write a model in one line — `return 'ham'`
> — and it is 86.6% accurate. It's also completely useless.
>
> That single statistic shaped every decision that followed: which metrics I
> trusted, how I tuned the model, and an entire phase dedicated to handling the
> imbalance. It's the reason I never quote accuracy on its own."

*Transition: "Before modelling anything, I wanted to know what actually separates the two classes."*

---

## Slide 5 — EDA · 45 seconds

> "Spam looks different before you even read it. On average it's about twice as
> long — it has to fit a pitch, a phone number and the small print. It uses far
> more capital letters and far more digits, because of prize amounts and
> shortcodes. And almost all the links in this dataset are in spam.
>
> Then I built word clouds for each class, and they barely overlap. Spam is
> *free, call, txt, claim, prize, win*. Ham is *ok, got, home, later, love*.
>
> That was the signal I needed: these two classes are separable by vocabulary
> alone. Which told me a bag-of-words approach would work, and I didn't need to
> reach for a neural network to solve this."

*Transition: "But a model can't read words. So the next step was turning text into numbers."*

---

## Slide 6 — Preprocessing and TF-IDF · 60 seconds

> "This is the most important function in the whole project, because everything
> downstream depends on what comes out of it.
>
> Seven steps. Lowercase everything, so 'FREE', 'Free' and 'free' are one
> signal, not three weak ones. Strip URLs and digits — and this one is subtle.
> Every spam link is a different string, so each becomes a unique token the
> model can never learn from. *Whether* there's a link is signal; *which* link
> it is, is noise. Then strip punctuation, split into words, drop stopwords like
> 'the' and 'and' that appear in everything, and finally lemmatize — reduce
> 'prizes' to 'prize' so they're one feature instead of two thin ones.
>
> Then TF-IDF: Term Frequency times Inverse Document Frequency. A word scores
> high if it's frequent *in this message* but rare *across the whole corpus*. So
> common words get damped and distinctive words like 'claim' get amplified.
>
> One detail I'm proud of: I included **bigrams** — adjacent word pairs. That
> means the model can learn 'free entry' and 'call now' as phrases, not just
> isolated words. You'll see that pay off in the live demo."

*Transition: "With numbers in hand, I could finally train something."*

---

## Slide 7 — Model comparison · 45 seconds

> "I didn't want to just pick Logistic Regression because it's the obvious
> choice. So I trained six algorithms on identical features and compared them
> across five metrics: Logistic Regression, Naive Bayes, Linear SVM, Random
> Forest, Gradient Boosting, and K-Nearest Neighbours.
>
> Logistic Regression and Linear SVM came out on top, essentially tied. **I chose
> Logistic Regression for three reasons.** It matched the best on accuracy. It
> returns true probabilities — an SVM gives you a distance from a boundary, not
> a probability, and both the confidence score and the sensitivity slider in my
> app depend on real probabilities. And every coefficient is directly readable,
> which is what makes the explanations on slide ten possible at all.
>
> KNN was in there deliberately as a control — distance barely means anything in
> twenty-thousand-dimensional sparse space, and its score confirmed that. The
> 'No Free Lunch' theorem in practice: the best model is the one that fits *this*
> problem, not the most powerful one available."

*Transition: "Then I tuned it properly."*

---

## Slide 8 — Tuning and imbalance · 50 seconds

> "On the left: instead of hand-picking settings, I used GridSearchCV — 36
> configurations, each cross-validated five times, so 180 model fits. And
> crucially I searched the *TF-IDF settings and the model settings together*,
> inside a single scikit-learn Pipeline.
>
> That Pipeline matters for a reason beyond convenience: it prevents data
> leakage. Because the vectorizer is refitted inside every cross-validation
> fold, vocabulary from the validation fold never leaks into training. If I'd
> vectorized everything up front, my scores would have been inflated.
>
> I also scored on F1 rather than accuracy — on imbalanced data, optimising
> accuracy would have tuned the model toward saying 'ham' more often, which is
> the opposite of what I wanted.
>
> On the right: I tested SMOTE, which generates synthetic spam examples by
> interpolating between real ones, rather than just duplicating them. Applied to
> the training set only — the test set has to stay at 13% spam, because that's
> reality.
>
> And the honest result: **balancing raised recall but cost precision, so the
> shipped model doesn't use SMOTE.** I think that's worth more than if it had
> worked, because now it's a measured decision rather than an assumption."

*Transition: "So — how well does it actually do?"*

---

## Slide 9 — Results · 75 seconds ⭐ *the most important slide*

> "On 1,115 messages the model had never seen: **98.03% accuracy, and a ROC-AUC
> of 99.30%.**
>
> But I want to break the confusion matrix down, because the headline number
> hides something.
>
> Of 960 legitimate messages, only **three** were wrongly flagged as spam.
> That's the error I care most about, and it's very low.
>
> But of 155 real spam messages, **nineteen got through**. So my spam recall is
> **87.7%**, not 98%.
>
> *(pause)*
>
> Why the gap? Because accuracy is computed over all 1,115 messages, and 960 of
> them are easy ham. The majority class drowns out the minority. **Accuracy
> flatters every model on data this imbalanced** — which is exactly why I
> reported precision, recall and F1 alongside it, and why quoting only the 98%
> would have been hiding the real weak spot.
>
> That's the number I'd work on next if this went to production."

> 💡 **Delivery note:** slow down here. The pause after "nineteen got through" is
> the most persuasive two seconds in the talk. Volunteering your own weakness
> before anyone asks reads as rigour, not failure — and it usually pre-empts the
> toughest question in the room.

*Transition: "The other thing I insisted on was that the model couldn't be a black box."*

---

## Slide 10 — Explainability · 50 seconds

> "Logistic Regression scores a message as its intercept plus the sum, across
> every word, of that word's TF-IDF value times its coefficient. Which means I
> can compute exactly how much each word contributed — not an estimate, the
> actual arithmetic.
>
> Here's what the model learned. On the spam side: 'txt' at minus 9.1, 'claim'
> at minus 7.7, then mobile, service, reply, prize, win, free, urgent, cash. On
> the ham side: 'ok', 'home', 'road', 'later', 'good'.
>
> **Nobody told it any of those words.** It rediscovered the entire vocabulary of
> SMS spam from 4,457 training examples — and the ham side is pure everyday
> conversation.
>
> I also implemented LIME in the notebook, which explains any model by perturbing
> the input and watching how predictions move. But the app uses the coefficient
> method, because for a linear model the exact answer is available in closed
> form — it's both more faithful than an approximation and about five hundred
> times faster. Knowing when *not* to reach for the fancier tool was part of the
> lesson."

*Transition: "And all of this runs in a live app — let me show you."*

---

## Slide 11 — The app + LIVE DEMO · 90 seconds ⭐

> "The last phase was making this usable by someone who doesn't write Python.
>
> Architecturally I split it in two: `app.py` is purely the interface, and
> `backend.py` holds all the inference logic with no Streamlit import at all.
> That means the backend is unit-tested without a browser — there are 30 tests
> behind it — and the same code could back an API tomorrow."

**→ Switch to the live app.**

**Demo, in this exact order:**

1. **Click "Ham — friendly chat" → Classify.**
   > "A normal message. Green, 99% confidence. Good."

2. **Click "Spam — prize draw" → Classify. Scroll to the explanation chart.**
   > "Now a real spam message from the dataset. Red, 93% confidence. And here's
   > the part I like — it tells me *why*: 'txt', 'win', 'free', 'entry'. And look
   > at this one — **'free entry'**, as a single feature. That's the bigram I
   > mentioned earlier, earning its place."

3. **Click "Phishing — account alert" → Classify.**
   > "Now watch this one. It comes out at about 45% — just under the line, so it's
   > called as ham, and the app *flags it as borderline*. It's genuinely
   > uncertain, and it says so instead of pretending."

4. **Move the sidebar slider to 0.35.**
   > "Now I'll make the filter more aggressive… and the same message flips to
   > spam. **The model didn't change. The policy did.** That's the trade-off on
   > the next slide."

5. *(If time) **Batch tab** → paste five messages → run.*
   > "And it scales — paste a list or upload a CSV, and you get every message
   > scored with a downloadable result."

> 💡 **If the app fails:** say *"I'll show you the screenshot instead"*, point at
> slide 11, and narrate the same four beats. Do not open a terminal.

---

## Slide 12 — The threshold trade-off · 40 seconds *(cut in the 5-minute version)*

> "That slider isn't a gimmick — it's the central engineering trade-off, and I
> measured it.
>
> At a 0.20 threshold the model catches 94% of spam, but wrongly flags 15
> legitimate messages. At the default 0.50, it catches 88% of spam and wrongly
> flags only 3.
>
> **There is no correct answer here — it's a business decision.** A marketing
> team might accept more false alarms; a hospital absolutely would not. So
> instead of burying that choice in a constant somewhere in the code, I exposed
> it as a slider and documented what each setting costs."

*Transition: "So, to wrap up."*

---

## Slide 13 — Conclusion · 40 seconds

> "In 45 days I built a complete NLP pipeline: 98% accurate, 99.3 ROC-AUC,
> chosen from six candidate models, tuned across 180 fits, explainable at the
> word level, deployed as a live web app, with 30 automated tests behind it.
>
> I'll be honest about the limits. It's trained on UK text messages from around
> 2005. It's bag-of-words, so it has no sense of word order. And it's beatable
> by anyone who writes 'F.R.E.E' with full stops.
>
> Which is exactly where I'd take it next: fine-tune a transformer like
> DistilBERT to capture context, retrain on a modern email corpus, add character
> n-grams to catch that obfuscation, and add monitoring — because spam evolves,
> and a model that's never retrained decays.
>
> Thank you. I'm happy to take questions."

---

# Q&A preparation

Answer in **one sentence first**, then elaborate if they look interested. If you
don't know, say *"I don't know — my instinct would be X, and I'd test it by Y."*
That answer scores better than a bluff every time.

### On the model

**"Why not use deep learning / BERT?"**
> Because it would be the wrong tool for this dataset. These are short messages
> separated almost perfectly by vocabulary, and my linear model already hits
> 99.3 ROC-AUC — BERT would cost a hundred times the compute for maybe a point,
> and I'd lose the exact explanations. I'd reach for it when word *order* starts
> to matter, like distinguishing "not free" from "free".

**"Why Logistic Regression over Linear SVM if SVM scored slightly higher?"**
> Probabilities. The SVM gives a distance from the decision boundary, not a
> calibrated probability, and my confidence display and threshold slider both
> need real probabilities. For a fraction of a point of F1, that wasn't a trade
> I wanted.

**"Is 98% accuracy actually good?"**
> Compared to the 86.6% you get by always guessing "ham", yes — but accuracy is
> the wrong headline. My spam recall is 87.7%, and that's the number that
> matters for a filter. I report both for exactly that reason.

**"Are you overfitting?"**
> I don't think so, and I checked rather than assumed: 5-fold cross-validation
> gave consistent scores with a small standard deviation, and the test set was
> held out from the start and only touched at evaluation. The gap between train
> and test accuracy is small.

### On the data and method

**"How do you know your preprocessing didn't destroy useful information?"**
> I don't know it for certain — I know the cleaned version outperformed the raw
> baseline. Stripping digits definitely loses something, since a shortcode is a
> genuine spam signal. If I revisited it I'd replace digits with a token like
> `<NUM>` rather than deleting them, and measure the difference.

**"Why did SMOTE not help, when the data is imbalanced?"**
> Because SMOTE on TF-IDF vectors is a bit strange — a synthetic point halfway
> between two spam vectors doesn't correspond to any real sentence. It did what
> it's supposed to: raised recall, lowered precision. But with only 3 false
> positives at baseline, trading precision away wasn't worth it here. It's a
> measured decision, which is the point of running the experiment.

**"What is data leakage and where could it have happened here?"**
> Leakage is when information from the test set influences training. The obvious
> place here was the vectorizer: if I'd fitted TF-IDF on the whole dataset before
> splitting, the IDF statistics would encode the test messages. I avoided it by
> putting the vectorizer inside a Pipeline, so it's refitted within each CV fold.

**"Why `random_state=3`?"**
> Reproducibility — it fixes the shuffle so anyone re-running the notebook gets
> my exact split. The specific value carries no meaning; the cross-validation
> results confirm the performance isn't an artefact of one lucky split.

### On deployment

**"How would this handle a million messages a day?"**
> The model itself is trivially fast — it's a sparse matrix multiply, roughly a
> millisecond per message, and it batches well. Streamlit is the bottleneck, not
> the model; for production I'd put `backend.py` behind FastAPI with the pipeline
> loaded once at start-up and scale horizontally.

**"What happens when spam changes?"**
> The model degrades — it only knows the 20,000 terms it was trained on, and new
> spam vocabulary is invisible to it. Production would need monitoring on live
> precision and recall, and periodic retraining on freshly labelled data.

**"Could someone deliberately get spam past it?"**
> Easily, and I'd rather say so. Write "F.R.E.E" or "pr1ze" and the tokens no
> longer match anything in the vocabulary. Character-level n-grams would help;
> real filters also use sender reputation, link analysis and header checks, none
> of which this model sees.

**"What was the hardest bug?"**
> The app loaded a model path relative to the working directory while the `.pkl`
> files sat somewhere else, so it crashed on start-up in deployment while working
> fine locally. The fix was resolving the path relative to the module file. The
> more dangerous one was version pinning — a model pickled by one scikit-learn
> version and loaded by another can silently change predictions with no error, so
> I pinned the version and added a runtime check that warns in the sidebar.

### The one to prepare hardest for

**"What would you do differently if you started over?"**
> Three things. I'd set up the train/test split and the evaluation harness on day
> one, before touching a model, so every experiment was comparable from the
> start. I'd version the data and the model together rather than discovering the
> pinning problem at deployment. And I'd have written the tests earlier — they
> found two real bugs in the app, and they'd have found them sooner.

---

## Timing card — tear this off

| Slide | Time | Cumulative | Cut in 5-min version? |
|---|---|---|---|
| 1 Title | 0:20 | 0:20 | keep |
| 2 Problem | 0:40 | 1:00 | keep |
| 3 Roadmap | 0:25 | 1:25 | **cut** |
| 4 Data | 0:40 | 2:05 | keep |
| 5 EDA | 0:45 | 2:50 | keep |
| 6 Preprocessing | 1:00 | 3:50 | trim to 0:40 |
| 7 Models | 0:45 | 4:35 | trim to 0:30 |
| 8 Tuning | 0:50 | 5:25 | trim to 0:30 |
| 9 **Results** | 1:15 | 6:40 | keep — never cut |
| 10 Explainability | 0:50 | 7:30 | trim to 0:35 |
| 11 **App + demo** | 1:30 | 9:00 | keep — never cut |
| 12 Threshold | 0:40 | 9:40 | **cut** |
| 13 Conclusion | 0:40 | 10:20 | keep |

**If you only have five minutes:** slides 1, 2, 4, 6, 9, 11, 13. Results and the
live demo are the whole talk; everything else is supporting evidence.
