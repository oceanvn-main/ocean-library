# How to Build "Intelligence" (and Consciousness): Notes from the AI Pioneers, 1950–1980

This file collects what the founding figures of AI said, in their papers, memos, reports and lectures between 1950 and 1980, about two questions:

1. **How could we build a system that is intelligent?**
2. **Could such a system be conscious, and what would that take?**

Each section gives the person's main claims in paraphrase, followed by the primary sources. Short quotations are kept to a minimum. A synthesis comes at the end (§9–§11).

> **About the references (checked 2026-10-21).** Volume, issue and page numbers were checked against online bibliographic records: publisher and DOI pages, IEEE Xplore, PubMed, JSTOR-derived copies, review listings in the *Journal of Symbolic Logic*, the authors' own bibliographies, and library catalogues. That check found and fixed five errors: Putnam 1960's pages, Searle 1980's page range, the number of levels in Marr & Poggio 1976, the publisher of the Ringle volume, and the original source of Simon's 1965 prediction.
>
> Five citations could only be partly confirmed. They are marked *[pages not confirmed against the original]* where they appear:
> - von Neumann 1951 (Hixon Symposium)
> - Shannon 1951/52 (maze machine)
> - Newell, Shaw & Simon 1959 (GPS): pages come from secondary records only
> - McCarthy 1959: sources give 75–91 or 77–84
> - Weizenbaum 1966
>
> §12 lists archives where most of these papers can be read.

---

## 0. Precursors (before 1950) that everyone in 1950–1980 built on

| Who | Work | Main idea for building intelligence |
|---|---|---|
| Warren McCulloch & Walter Pitts | "A Logical Calculus of the Ideas Immanent in Nervous Activity", *Bull. Math. Biophysics* 5:115–133 (1943) | Idealised neurons are threshold logic units, and networks of them can compute any logical function. The brain can therefore be treated as a computing machine. |
| Kenneth Craik | *The Nature of Explanation*, Cambridge UP (1943) | Thought works by building **internal "small-scale models"** of the world and running them to predict what will happen. This is the origin of the "world model" idea. |
| Norbert Wiener | *Cybernetics: or Control and Communication in the Animal and the Machine*, MIT Press/Wiley (1948) | Purposeful behaviour comes from **feedback**: comparing goal and state, then correcting. Animals and machines share the same control-and-communication principles. |
| Claude Shannon | "A Mathematical Theory of Communication", *Bell Syst. Tech. J.* 27:379–423, 623–656 (1948) | Information can be quantified independently of meaning. This gave AI its vocabulary of bits, channels, redundancy and statistical structure. |
| Alan Turing | Lecture to the London Mathematical Society, 20 Feb 1947 (in Copeland 2004) | Asks for "a machine that can learn from experience" and suggests a computer could alter its own instructions. |

---

## 1. Alan Turing (1912–1954): an operational test, learning machines, and the consciousness objection

### 1.1 "Intelligent Machinery" (NPL internal report, 1948)
- Introduces **"unorganised machines"**: randomly connected networks of simple NAND-like units. These are an early form of neural networks.
- Proposes **organising them by training**, using "pleasure" and "pain" inputs (an early form of reinforcement) together with outside "interference".
- Argues that intelligence needs both **discipline** (a universal machine following rules) and **initiative** (searching, guessing, going beyond the rules).
- Lists the areas where machine intelligence could first be shown: games (chess), languages, translation, cryptography, mathematics.
- *Source:* Turing, A. M. (1948). *Intelligent Machinery*. NPL Report. Published in Evans & Robertson (eds.), *Cybernetics* (1968), and in *Machine Intelligence 5* (1969), pp. 3–23. Reprinted in Copeland (2004).

### 1.2 "Computing Machinery and Intelligence" (*Mind*, 1950)
- **Replaces "Can machines think?"** with the **imitation game**, an operational, behavioural test. The question shifts from what a machine *is* to what it *does* in open-ended conversation.
- Predicts that by about 2000, a machine with about 10⁹ bits of storage would play the game well enough that an average interrogator would have at most a 70% chance of a correct identification after five minutes.
- Answers nine objections. Three matter most for design:
  - **The argument from consciousness** (from Jefferson, 1949): if we demand proof that a machine *feels*, we must demand the same of other people, and that leads to solipsism. Turing suggests the "polite convention" that everyone who behaves this way thinks. He grants that consciousness is mysterious but says that mystery need not be solved before we can answer whether machines can pass the test.
  - **Lady Lovelace's objection** (a machine only does what we tell it): learning machines can surprise their makers.
  - **Arguments from various disabilities** ("a machine can never do X"): these are mostly unexamined inductions from the machines we have seen so far.
- **The design recipe (§7, "Learning Machines"):** do not try to program an adult mind. Build a **"child machine"** and **educate** it. The parts are: (a) the initial state of the mind (innate structure), (b) education, and (c) other experience. Use **rewards and punishments**, include a **random element** in the learner, and expect the teacher to be largely ignorant of what is going on inside. Turing draws an explicit analogy with evolution: the child machine's structure corresponds to hereditary material, changes to it correspond to mutation, and the experimenter's judgement corresponds to natural selection.
- *Source:* Turing, A. M. (1950). Computing Machinery and Intelligence. *Mind* 59(236):433–460.

### 1.3 Radio talks and lectures (1951–1952)
- "**Intelligent Machinery, A Heretical Theory**" (lecture, c. 1951): machines that learn from experience could eventually outstrip their makers. Turing speculates that at some stage we should expect the machines to "take control".
- "**Can Digital Computers Think?**" (BBC Third Programme, 15 May 1951): a universal machine with suitable programming can mimic the brain. The limits are storage and speed, not principle.
- "**Can Automatic Calculating Machines Be Said to Think?**" (BBC broadcast discussion with Newman, Braithwaite and Jefferson, 14 Jan 1952): defends the behavioural criterion against the demand for a felt inner life.
- *Source:* all three in Copeland, B. J. (ed.) (2004). *The Essential Turing*. Oxford UP.

**Turing's lessons for building intelligence:** define the target behaviourally, get there by *learning* rather than hand-coding, start from a well-chosen innate structure, include randomness, and leave consciousness as an open problem rather than a precondition.

---

## 2. Claude Shannon (1916–2001): search, evaluation, and machines that learn mazes

- **"Programming a Computer for Playing Chess" (1950).** This was the first technical blueprint for game-playing AI:
  - Represent positions symbolically and **search the game tree with minimax**.
  - Use a heuristic **evaluation function** (material, mobility, pawn structure) at the search frontier.
  - **Type A strategy:** brute-force search to a fixed depth. **Type B strategy:** selective search of only "plausible" moves, as human players do. The Type A/Type B split became the long-running debate between brute force and heuristic selectivity.
  - Shannon notes that playing chess well would show something like thinking, while being frank that a chess program is not general intelligence.
  - *Source:* Shannon, C. E. (1950). Programming a Computer for Playing Chess. *Philosophical Magazine* Ser. 7, 41(314):256–275.
- **Theseus (1950).** An electromechanical mouse that learns a maze by trial and error, remembers the solution and adapts when the maze changes. It was a physical demonstration of learning and memory in a machine.
  - *Source:* Shannon, C. E. (1952). Presentation of a Maze-Solving Machine. In H. von Foerster (ed.), *Cybernetics: Transactions of the Eighth Conference, March 15–16, 1951*. Josiah Macy, Jr. Foundation, pp. 173–180 *[pages not confirmed against the original]*. Reprinted in C. Pias (ed.) (2016), *Cybernetics: The Macy Conferences 1946–1953. The Complete Transactions*, diaphanes, pp. 474–479. Some sources title it "…Maze-Solving Mouse".
- **"Computers and Automata" (1953).** Surveys the field and lists the open problems: machines that improve themselves, that handle abstraction, that do "creative" work, and that reproduce themselves.
  - *Source:* Shannon, C. E. (1953). Computers and Automata. *Proceedings of the IRE* 41(10):1234–1241.
- **Automata Studies (1956), edited with McCarthy.** Collected the theory of finite automata, neural nets and reliable computation (von Neumann, Kleene, Minsky and others).
  - *Source:* Shannon, C. E. & McCarthy, J. (eds.) (1956). *Automata Studies*. Annals of Mathematics Studies 34, Princeton UP.

**Shannon's lessons:** intelligence as **search plus heuristic evaluation**, with **learning from trial and error** as a separate mechanism. Also, test ideas on small, well-defined domains first.

---

## 3. The Dartmouth Proposal (1955): the founding memo of "Artificial Intelligence"

Written by **John McCarthy, Marvin Minsky, Nathaniel Rochester and Claude Shannon**. The memo coined the term "artificial intelligence".

- **Core conjecture:** every aspect of learning, and every other feature of intelligence, can in principle be described so precisely that a machine can be made to simulate it.
- **Seven research problems** are listed:
  1. Automatic computers: the issue is programming, not hardware.
  2. How a computer can be programmed to **use a language**.
  3. **Neuron nets**: how hypothetical neurons can be arranged to form concepts.
  4. Theory of the **size of a calculation**: efficiency and complexity.
  5. **Self-improvement**: machines that improve themselves.
  6. **Abstractions**: forming concepts from sensory data.
  7. **Randomness and creativity**: controlled randomness as the difference between creative and merely competent thinking.
- Individual research plans: Shannon (information theory applied to brain models), Minsky (a learning machine with an internal model of its environment), Rochester (originality via randomness), McCarthy (an artificial language for reasoning and self-reference).
- *Source:* McCarthy, J., Minsky, M. L., Rochester, N. & Shannon, C. E. (1955). *A Proposal for the Dartmouth Summer Research Project on Artificial Intelligence*. Dated 31 Aug 1955. Reprinted in *AI Magazine* 27(4):12–14 (2006).

---

## 4. Cybernetics and brain-inspired engineering (1948–1960)

### 4.1 John von Neumann (1903–1957): the logic of automata, reliability, and the brain
- **"The General and Logical Theory of Automata" (Hixon Symposium, 1948; published 1951).** Compares computers and nervous systems. Argues that a full description of complex behaviour (for example visual analogy) may be no simpler than the network that produces it. Starts the theory of **self-reproducing automata**.
  - *Source:* von Neumann, J. (1951). In L. A. Jeffress (ed.), *Cerebral Mechanisms in Behavior: The Hixon Symposium*, Wiley, pp. 1–41. It is the volume's opening chapter. *[Pages not confirmed against the original.]* Reprinted in A. H. Taub (ed.), *John von Neumann: Collected Works*, Vol. 5, Pergamon (1963), pp. 288–328.
- **"Probabilistic Logics and the Synthesis of Reliable Organisms from Unreliable Components" (1956).** Shows that **redundancy and multiplexing** let a reliable system be built from noisy parts, which is how the brain must work.
  - *Source:* in Shannon & McCarthy (eds.), *Automata Studies* (1956), pp. 43–98.
- ***The Computer and the Brain* (Silliman Lectures, written 1955–56, published 1958).** The brain is massively parallel, slow and low-precision, and partly analogue/statistical. Its "language" is probably **not the language of mathematics**: the logic and arithmetic it actually uses may be very different from ours.
  - *Source:* von Neumann, J. (1958). *The Computer and the Brain*. Yale UP.
- ***Theory of Self-Reproducing Automata*** (completed and edited by A. W. Burks, 1966, Univ. of Illinois Press). Cellular automata that build copies of themselves. The point is that complexity can grow through construction.

### 4.2 Norbert Wiener (1894–1964): feedback, purpose, and the control problem
- *The Human Use of Human Beings* (1950): society and machines as communication-and-feedback systems.
- **"Some Moral and Technical Consequences of Automation" (1960).** Machines that learn can act faster than we can correct them. If we use a machine whose operation we cannot efficiently interfere with, we had better be sure that the purpose put into it is the one we really want. This is the earliest clear statement of what is now called the **alignment problem**.
  - *Source:* Wiener, N. (1960). *Science* 131(3410):1355–1358.
- *God & Golem, Inc.* (1964), MIT Press: learning machines, self-reproducing machines, and the ethics of both.

### 4.3 W. Ross Ashby (1903–1972): adaptation as ultrastability
- ***Design for a Brain* (1952).** The **Homeostat**: a machine that keeps its essential variables within limits by randomly re-wiring its own parameters until it finds a stable configuration (**ultrastability**). On this view, adaptive behaviour is the system's search for equilibrium with its environment.
  - *Source:* Ashby, W. R. (1952). *Design for a Brain*. Chapman & Hall (2nd ed. 1960).
- ***An Introduction to Cybernetics* (1956).** The **Law of Requisite Variety**: a controller must have at least as much variety as the disturbances it must counter. An intelligent system has to be internally as rich as its world.
  - *Source:* Ashby, W. R. (1956). Chapman & Hall.

### 4.4 W. Grey Walter (1910–1977): minimal embodied agents
- **Machina speculatrix** "tortoises" (Elmer and Elsie): two "neurons", a light sensor and a touch sensor. Their behaviour was surprisingly lifelike: seeking light, avoiding obstacles, returning to recharge. Complex behaviour emerges from simple circuits interacting with the environment.
  - *Sources:* Walter, W. G. (1950, May). An Imitation of Life. *Scientific American* 182(5):42–45; (1951, Aug). A Machine That Learns. *Scientific American* 185(2):60–63; (1953). *The Living Brain*. Duckworth.

**Cybernetic lessons:** intelligence is **closed-loop, adaptive and embodied**. Reliability comes from redundancy, a controller must be internally rich enough for its world, and a learning machine's goals must be specified with great care.

---

## 5. The symbolic school: search, heuristics, logic and knowledge

### 5.1 Allen Newell (1927–1992) & Herbert A. Simon (1916–2001)
- **Logic Theorist (1956).** The first program built specifically to do something "intelligent": it proved theorems from *Principia Mathematica* using **heuristic search** rather than exhaustive enumeration.
  - *Source:* Newell, A. & Simon, H. A. (1956). The Logic Theory Machine: A Complex Information Processing System. *IRE Trans. Information Theory* IT-2(3):61–79.
- **"Elements of a Theory of Human Problem Solving" (1958).** A program is a *theory of the mind*: if it solves problems the way humans do, it explains human thinking. This founded **cognitive simulation**.
  - *Source:* Newell, A., Shaw, J. C. & Simon, H. A. (1958). *Psychological Review* 65(3):151–166.
- **"Heuristic Problem Solving: The Next Advance in Operations Research" (1958).** Contains the famous predictions: within ten years a computer would be world chess champion, would discover an important new mathematical theorem, and most psychological theories would take the form of programs.
  - *Source:* Simon, H. A. & Newell, A. (1958). *Operations Research* 6(1):1–10.
- **General Problem Solver (GPS, 1959).** **Means-ends analysis**: compare the current state with the goal, find the difference, and apply an operator that reduces it, recursing on subgoals. GPS separates *general* problem-solving methods from *task-specific* knowledge.
  - *Source:* Newell, A., Shaw, J. C. & Simon, H. A. (1959). Report on a General Problem-Solving Program. *Proc. Int. Conf. on Information Processing* (UNESCO, Paris), pp. 256–264. *[Pages come from secondary records only.]*
- ***Human Problem Solving* (1972).** **Problem spaces**, **production systems**, and the analysis of think-aloud protocols.
  - *Source:* Newell, A. & Simon, H. A. (1972). Prentice-Hall.
- **Turing Award Lecture (1975, published 1976):** the two central hypotheses of symbolic AI.
  - **Physical Symbol System Hypothesis:** a physical symbol system has the necessary and sufficient means for general intelligent action.
  - **Heuristic Search Hypothesis:** symbol systems solve problems by generating candidate solutions and progressively modifying them, that is, by heuristic search.
  - *Source:* Newell, A. & Simon, H. A. (1976). Computer Science as Empirical Inquiry: Symbols and Search. *Communications of the ACM* 19(3):113–126.
- **Newell, "Physical Symbol Systems" (1980)** gives the full formal treatment. **Newell, "The Knowledge Level"** (AAAI presidential address, 1980; published 1981/82) describes a system by its goals and knowledge plus the **principle of rationality**, independently of how it is implemented.
  - *Sources:* Newell, A. (1980). *Cognitive Science* 4(2):135–183; Newell, A. (1982). *Artificial Intelligence* 18(1):87–127.
- **Simon, *The Sciences of the Artificial* (1969).** **Bounded rationality** and **satisficing**. The parable of the **ant on the beach**: the ant's apparently complex path mostly reflects the complexity of the beach, not of the ant. Hierarchical, "nearly decomposable" systems evolve and get designed much faster than flat ones.
  - *Source:* Simon, H. A. (1969). MIT Press.

### 5.2 John McCarthy (1927–2011): common sense, logic and the frame problem
- **LISP (1958–60):** symbolic list processing, recursion, and code as data. It became the working language of AI.
  - *Source:* McCarthy, J. (1960). Recursive Functions of Symbolic Expressions and Their Computation by Machine, Part I. *CACM* 3(4):184–195.
- **"Programs with Common Sense" (1958/59): the "Advice Taker".** A program should represent its knowledge as **declarative sentences in formal logic**, so that it can be improved simply by *telling* it new facts rather than reprogramming it. Common sense means drawing the immediate consequences of what you have been told.
  - *Source:* McCarthy, J. (1959). In *Mechanisation of Thought Processes: Proc. Symposium at the NPL, Teddington, Nov 1958*. HMSO, pp. 75–91. *[Most records give 75–91; at least one gives 77–84.]*
- **McCarthy & Hayes (1969):**
  - Distinguishes **epistemological adequacy** (can the representation express the facts?) from **heuristic adequacy** (can the reasoning use them efficiently?).
  - Introduces the **situation calculus** and names the **frame problem**: how to represent compactly what does *not* change when an action happens.
  - *Source:* McCarthy, J. & Hayes, P. J. (1969). Some Philosophical Problems from the Standpoint of Artificial Intelligence. *Machine Intelligence 4*, Edinburgh UP, pp. 463–502.
- **"Epistemological Problems of Artificial Intelligence" (1977)** and **"Circumscription" (1980):** formalising **non-monotonic reasoning**, that is, jumping to defaults that can be withdrawn.
  - *Sources:* McCarthy, J. (1977). *Proc. IJCAI-77*, pp. 1038–1044; McCarthy, J. (1980). *Artificial Intelligence* 13(1–2):27–39.
- **"Ascribing Mental Qualities to Machines" (1979):** see §8.3.

### 5.3 Marvin Minsky (1927–2016): from neural nets to frames to a society of mind
- **SNARC (1951)** and his **Princeton PhD (1954)**, *Theory of Neural-Analog Reinforcement Systems*: an early randomly wired reinforcement learner built from vacuum tubes.
- **"Steps Toward Artificial Intelligence" (1961).** The field's first programmatic survey. It divides the problem into five parts:
  1. **Search**
  2. **Pattern recognition**
  3. **Learning**
  4. **Planning**
  5. **Induction**

  It names the **credit assignment problem**: in a long chain of actions, which decisions deserve credit for success? It also argues that **hierarchy and planning** are needed to escape combinatorial explosion.
  - *Source:* Minsky, M. (1961). *Proceedings of the IRE* 49(1):8–30.
- **"Matter, Mind and Models" (1965):** see §8.2.
- **Computation: Finite and Infinite Machines (1967).** The preface predicts that within a generation the problem of creating AI would be substantially solved.
  - *Source:* Minsky, M. (1967). Prentice-Hall.
- **Perceptrons (1969), with Seymour Papert.** Proves the limits of single-layer perceptrons: they cannot compute connectedness or parity under the locality and order restrictions studied. The book is widely blamed for diverting funding from neural nets, although it explicitly left multi-layer networks open.
  - *Source:* Minsky, M. & Papert, S. (1969). *Perceptrons*. MIT Press.
- **"A Framework for Representing Knowledge" (1974, MIT AI Memo 306).** **Frames** are stereotyped data structures with slots and **default values**, used to recognise situations and fill gaps through expectation.
- **"K-Lines: A Theory of Memory" (1980).** Memory works by re-activating the set of "agents" that were active when a problem was solved.
  - *Source:* Minsky, M. (1980). *Cognitive Science* 4(2):117–133.
- Ideas developed with Papert in the 1970s led to ***The Society of Mind*** (published 1986, outside this window). Intelligence emerges from many mindless agents.

### 5.4 Knowledge-based systems (1965–1980)
- **Edward Feigenbaum, Bruce Buchanan, Joshua Lederberg: DENDRAL (1965 onward)** inferred molecular structure from mass spectra. Feigenbaum's **Knowledge Principle**: a program's power comes mainly from its **specific domain knowledge**, not its general inference method.
  - *Sources:* Buchanan, B., Sutherland, G. & Feigenbaum, E. (1969). Heuristic DENDRAL. *Machine Intelligence 4*, pp. 209–254; Feigenbaum, E. A. (1977). The Art of Artificial Intelligence: Themes and Case Studies of Knowledge Engineering. *Proc. IJCAI-77*, pp. 1014–1029.
- **Terry Winograd: SHRDLU (1968–70).** Language understanding in a **blocks world** that integrates syntax, semantics and reasoning. It worked impressively, but only inside a micro-world.
  - *Source:* Winograd, T. (1972). Understanding Natural Language. *Cognitive Psychology* 3(1):1–191. Also issued as a book by Academic Press (1972).
- **Roger Schank & Robert Abelson: scripts (1977).** Understanding a story requires stereotyped knowledge of event sequences (the "restaurant script").
  - *Source:* Schank, R. & Abelson, R. (1977). *Scripts, Plans, Goals and Understanding*. Erlbaum.
- **Edward Feigenbaum & Julian Feldman (eds.), *Computers and Thought* (1963), McGraw-Hill.** The first AI anthology. It reprints Turing (1950), Samuel, Newell–Shaw–Simon, Minsky (1961), Selfridge and others.

**Symbolic lessons:** intelligence as **search through problem spaces guided by heuristics**, operating over **explicit symbolic representations** of knowledge, with common-sense defaults. Power comes from *knowledge*, not just from general methods.

---

## 6. The learning and connectionist school

- **Frank Rosenblatt (1928–1971): the Perceptron (1958).** A probabilistic, brain-inspired network whose connection weights are **learned from examples** by error correction. Rosenblatt argued that *statistical separability* rather than symbolic logic is the right language for brain models.
  - *Sources:* Rosenblatt, F. (1958). The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain. *Psychological Review* 65(6):386–408; (1962). *Principles of Neurodynamics*. Spartan Books.
- **Oliver Selfridge (1926–2008): Pandemonium (1958/59).** A hierarchy of "demons", each detecting a feature and "shouting" in proportion to its evidence, with decision demons listening to the loudest. Weights and features adapt with experience. This is an early form of **hierarchical feature learning and parallel distributed processing**.
  - *Source:* Selfridge, O. G. (1959). Pandemonium: A Paradigm for Learning. In *Mechanisation of Thought Processes*, HMSO, pp. 511–529.
- **Arthur Samuel (1901–1990): checkers (1959).** A program that **improves through self-play**. It uses rote learning of positions plus an evaluation polynomial whose weights are adjusted from the difference between successive evaluations, an early form of **temporal-difference learning**. It eventually beat its creator. Samuel argued that programming computers to learn from experience could remove the need for much detailed programming effort.
  - *Sources:* Samuel, A. L. (1959). Some Studies in Machine Learning Using the Game of Checkers. *IBM J. Research & Development* 3(3):210–229; (1967) Part II. *IBM J. R&D* 11(6):601–617.
- **Ray Solomonoff (1926–2009): algorithmic induction (1956–1964).** Prediction as finding the **shortest program** that generates the observed data, weighted by a universal prior. This gave a mathematical ideal for learning from data.
  - *Source:* Solomonoff, R. J. (1964). A Formal Theory of Inductive Inference, Parts I & II. *Information and Control* 7(1):1–22; 7(2):224–254.
- **John Holland (1929–2015): genetic algorithms (1975).** Adaptation as **evolutionary search** over a population of encoded structures, using selection, crossover and mutation. The schema theorem explains why this works.
  - *Source:* Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. Univ. of Michigan Press.
- **Paul Werbos: backpropagation (1974).** Gradient-based training of multi-layer networks through reverse-mode differentiation. It was largely ignored until the 1980s.
  - *Source:* Werbos, P. J. (1974). *Beyond Regression: New Tools for Prediction and Analysis in the Behavioral Sciences*. PhD thesis, Harvard.
- **Kunihiko Fukushima: Neocognitron (1980).** A hierarchical, **shift-invariant** network of alternating feature-extraction and pooling layers, inspired by Hubel and Wiesel's work on visual cortex. It is the direct ancestor of CNNs.
  - *Source:* Fukushima, K. (1980). *Biological Cybernetics* 36(4):193–202.
- **I. J. Good (1916–2009): "Speculations Concerning the First Ultraintelligent Machine" (1965).** An ultraintelligent machine could design even better machines, leading to an **"intelligence explosion"**. The first such machine would then be the last invention people need ever make, *provided* it is docile enough to tell us how to keep it under control.
  - *Source:* Good, I. J. (1965). *Advances in Computers* 6:31–88.

**Learning lessons:** do not program the knowledge; **learn it from data, self-play or evolution**. Use hierarchical feature detectors, error-driven weight updates and a bias toward simple explanations. Turing (1950) had already pointed in this direction.

---

## 7. Critics and limits (1961–1980)

| Critic | Work | Core argument | What it implies for building intelligence |
|---|---|---|---|
| **John Lucas** | "Minds, Machines and Gödel", *Philosophy* 36(137):112–127 (1961) | By Gödel's theorem, a human can see the truth of a formal system's Gödel sentence, which the system cannot prove, so minds are not machines. | Most logicians rejected the argument: humans are not known to be consistent, nor to know their own formal system. Turing had already answered this ("the mathematical objection", 1950). |
| **Hubert Dreyfus** | *Alchemy and Artificial Intelligence*, RAND Paper P-3244 (1965); *What Computers Can't Do*, Harper & Row (1972) | Human expertise rests on **embodied, situated, tacit know-how** and a background of relevance that cannot be fully written down as explicit rules. Symbolic AI's early success in micro-worlds would not scale. | Intelligence may require **embodiment, context and learned (not stated) skill**. Later embodied cognition and connectionism partly vindicated this. |
| **Joseph Weizenbaum** | "ELIZA—A Computer Program for the Study of Natural Language Communication Between Man and Machine", *CACM* 9(1):36–45 (1966) *[pages not confirmed against the original]*; *Computer Power and Human Reason*, W. H. Freeman (1976) | ELIZA showed that people readily project understanding onto shallow pattern-matching (now called the "ELIZA effect"). Some tasks, such as judgement, care and therapy, *should not* be handed to machines even if they could be. | Passing as intelligent in conversation is not evidence of understanding. Ethics belongs in the design question. |
| **Sir James Lighthill** | "Artificial Intelligence: A General Survey", in *Artificial Intelligence: A Paper Symposium*, Science Research Council, UK (1973) | **Combinatorial explosion**: methods that work on toy problems fail on real-world scale. This led to the first "AI winter" in the UK. | Scaling and generality are the real problems, not the toy demonstrations. |
| **John Searle** | "Minds, Brains, and Programs", *Behavioral and Brain Sciences* 3(3):417–424 (1980). The target article is pp. 417–424; with the open peer commentaries and Searle's response the whole package runs to p. 457. | **The Chinese Room**: manipulating symbols by their syntax is not sufficient for semantics or understanding. "Strong AI" (a program that by itself *has* a mind) is false. Mind needs the **causal powers** of brains, or their equivalent. | See §8.5. |

---

## 8. Consciousness: what the pioneers actually said

Most AI researchers in this period **deliberately sidestepped** consciousness and worked on behaviour and competence. The explicit positions are these.

### 8.1 Jefferson (1949) vs. Turing (1950)
- **Geoffrey Jefferson's Lister Oration** (a neurosurgeon) set the anti-mechanist bar: a machine would equal the brain only if it wrote a sonnet *because* of thoughts and emotions it *felt*, not by chance arrangement of symbols, and *knew* that it had written it.
  - *Source:* Jefferson, G. (1949). The Mind of Mechanical Man. *British Medical Journal* 1(4616):1105–1110.
- **Turing's reply:** demanding first-person proof leads to solipsism, so we should apply the same "polite convention" we apply to other people. Turing admits the problem is real ("I do not wish to give the impression that I think there is no mystery about consciousness") but says it does not need solving to answer the behavioural question. *(Turing 1950, §6(4).)*

### 8.2 Minsky (1965): consciousness and free will as features of self-models
- A machine (or person) that answers questions about the world uses an internal **model of the world (W\*)**. To answer questions about itself it needs a **model of itself (M\*)**, and of itself inside W\*.
- Such self-models are necessarily **crude and divided**. The model of oneself tends to split into a "physical" part and a "mental" part, which explains why intelligent systems naturally come to believe in **mind–body dualism** and **free will**. These beliefs are *artefacts of how self-models are built*, not evidence of non-physical minds.
- **Design implication:** a system that reports on its own mental states must contain an explicit, simplified **self-model**. Puzzling claims about "consciousness" are to be expected from it.
- *Source:* Minsky, M. (1965). Matter, Mind and Models. *Proc. IFIP Congress 1965*, Vol. 1, pp. 45–49. Reprinted in Minsky (ed.), *Semantic Information Processing*, MIT Press (1968).

### 8.3 McCarthy (1979): ascribing beliefs, and introspection as a design requirement
- It is legitimate and *useful* to ascribe **beliefs, goals, knowledge and intentions** to machines, even to a thermostat ("it believes the room is too cold"), when doing so helps predict and explain behaviour.
- Mental qualities come in **degrees and kinds**. A machine can have some, such as beliefs and goals, without others, such as feelings.
- **Introspection and self-knowledge** are capabilities to be *designed in*: a robot should be able to observe its own reasoning, know what it knows and doesn't know, and report on its goals. This is McCarthy's engineering route to the functional side of consciousness. He expanded on it in "Making Robots Conscious of Their Mental States" (1995), outside this window.
- *Source:* McCarthy, J. (1979). Ascribing Mental Qualities to Machines. In M. Ringle (ed.), *Philosophical Perspectives in Artificial Intelligence*. Harvester Press (Brighton) / Humanities Press (Atlantic Highlands, NJ), pp. 161–195. Also issued as Stanford AI Memo AIM-326.

### 8.4 Functionalism: the philosophical basis for machine minds
- **Hilary Putnam (1960, 1967):** mental states are defined by their **functional (causal-computational) role**, not by the material that realises them. This **multiple realisability** means a properly organised machine could have mental states. Functionalism became the default philosophy of mind behind AI and cognitive science.
  - *Sources:* Putnam, H. (1960). Minds and Machines. In S. Hook (ed.), *Dimensions of Mind*, NYU Press, pp. 148–179; Putnam, H. (1967). Psychological Predicates. In W. H. Capitan & D. D. Merrill (eds.), *Art, Mind, and Religion*, Univ. of Pittsburgh Press, pp. 37–48. The 1967 paper was later reprinted as "The Nature of Mental States".
- **Jerry Fodor, *The Language of Thought* (1975), Crowell:** thinking is computation over mental representations that have a language-like syntax.
- **Daniel Dennett, *Brainstorms* (1978), Bradford Books:** the **intentional stance**. Treat a system as a rational agent with beliefs and desires whenever doing so predicts its behaviour. Includes "Toward a Cognitive Theory of Consciousness" and "Why You Can't Make a Computer That Feels Pain".

### 8.5 The case for a hard limit
- **Thomas Nagel (1974):** an organism is conscious if there is **something it is like to be** that organism. This subjective character escapes every objective, functional or physical description known so far. Functional or behavioural success therefore would not establish consciousness.
  - *Source:* Nagel, T. (1974). What Is It Like to Be a Bat? *Philosophical Review* 83(4):435–450.
- **John Searle (1980):** running the right program is never *by itself* sufficient for understanding (intentionality). Any artefact with a mind would need causal powers at least equivalent to those of brains. Searle accepts "weak AI" (computers as tools for studying the mind) and rejects "strong AI".
  - *Source:* Searle, J. R. (1980). Minds, Brains, and Programs. *Behavioral and Brain Sciences* 3(3):417–424.

### 8.6 Self-reference and "strange loops"
- **Douglas Hofstadter, *Gödel, Escher, Bach* (1979), Basic Books.** Consciousness and the "I" arise from **self-referential, level-crossing loops** in a symbol system, where high-level patterns represent and influence the system that produces them. This is a constructive and optimistic answer to the Gödelian arguments of Lucas.

### 8.7 Summary of positions on machine consciousness (1950–1980)

| Position | Proponents | What you would need to build |
|---|---|---|
| **Behavioural / agnostic**: judge by behaviour, set the mystery aside | Turing (1950) | A system that behaves indistinguishably in open conversation |
| **Self-model**: "consciousness" talk comes from self-modelling | Minsky (1965); Craik's internal models before him | An explicit, simplified world model and self model |
| **Engineered introspection** | McCarthy (1979) | Mechanisms to observe and report one's own beliefs, goals and reasoning |
| **Functionalism / intentional stance** | Putnam (1960, 1967); Fodor (1975); Dennett (1978) | The right functional organisation, in any substrate |
| **Self-reference / strange loops** | Hofstadter (1979) | Rich self-referential symbol structures |
| **Subjective character is not captured functionally** | Nagel (1974) | Unknown; no current theory says |
| **Biological causal powers are required** | Jefferson (1949); Searle (1980) | Something with brain-equivalent causal powers, not just a program |
| **Impossible for formal systems** | Lucas (1961) | Not possible, though the argument is widely judged unsound |

---

## 9. Synthesis: recurring design principles for an "intelligent" system (1950–1980)

| # | Principle | Who stated it | Primary source |
|---|---|---|---|
| 1 | **Define intelligence operationally**, by performance on open-ended tasks rather than by inner essence | Turing | Turing 1950 |
| 2 | **Learn, don't hand-code**: a child machine plus education, with reward and punishment | Turing; Samuel; Rosenblatt | Turing 1948, 1950; Samuel 1959; Rosenblatt 1958 |
| 3 | **Search plus heuristic evaluation** through a space of possibilities | Shannon; Newell & Simon | Shannon 1950; Newell & Simon 1956, 1976 |
| 4 | **Explicit symbolic representation** of knowledge: the physical symbol system | Newell & Simon; McCarthy | Newell & Simon 1976; McCarthy 1959 |
| 5 | **Separate general methods from domain knowledge**, and power comes mostly from knowledge | Newell–Shaw–Simon (GPS); Feigenbaum | NSS 1959; Feigenbaum 1977 |
| 6 | **Common sense and defaults**: frames, scripts, non-monotonic logic, and the frame problem | McCarthy; Minsky; Schank | McCarthy & Hayes 1969; Minsky 1974; McCarthy 1980 |
| 7 | **Feedback and adaptation**: closed-loop, goal-seeking, ultrastable | Wiener; Ashby; Grey Walter | Wiener 1948; Ashby 1952 |
| 8 | **Reliability from unreliable parts**: redundancy, parallelism, low precision | von Neumann | von Neumann 1956, 1958 |
| 9 | **Internal models of the world and of oneself** | Craik; Minsky; Dartmouth (Minsky's plan) | Craik 1943; Minsky 1965 |
| 10 | **Hierarchy and decomposition** to escape combinatorial explosion | Minsky; Simon | Minsky 1961; Simon 1969 |
| 11 | **Many simple agents produce intelligence collectively** | Selfridge; Minsky | Selfridge 1959; Minsky 1980 |
| 12 | **Induction as compression**: prefer the simplest explanation | Solomonoff | Solomonoff 1964 |
| 13 | **Controlled randomness** as a source of creativity | Turing; Rochester (Dartmouth) | Turing 1950; Dartmouth 1955 |
| 14 | **Self-improvement**, possibly leading to an intelligence explosion | Dartmouth; Good | Dartmouth 1955; Good 1965 |
| 15 | **Environment shapes apparent complexity** (the ant on the beach) | Simon | Simon 1969 |
| 16 | **Embodiment and tacit skill** matter | Dreyfus; Grey Walter | Dreyfus 1972; Walter 1950 |
| 17 | **Specify the purpose carefully**: the control or alignment problem | Wiener; Good; Turing | Wiener 1960; Good 1965; Turing c. 1951 |
| 18 | **Separate levels of analysis**: what is computed, by what algorithm, with what mechanisms, in what hardware | Marr & Poggio; Newell (knowledge level) | Marr & Poggio 1976 (MIT AI Memo 357); Newell 1982 |

Marr & Poggio's levels come from: Marr, D. & Poggio, T. (1976). *From Understanding Computation to Understanding Neural Circuitry*. MIT AI Memo 357 (May 1976). The memo proposes **four** nearly independent levels: the computation, the algorithm, the mechanisms, and the hardware. Marr's *Vision* (1982) later condensed these into the better-known **three** levels: computational theory, representation and algorithm, and hardware implementation.

---

## 10. Predictions vs. outcomes: a calibration note

| Prediction | Source | Outcome |
|---|---|---|
| A machine would be world chess champion within 10 years (by 1968) | Simon & Newell 1958 | Took about 40 years (Deep Blue, 1997), and by brute-force Type A search more than human-like Type B |
| Machines would be capable, within 20 years, of doing any work a man can do | Simon, *The Shape of Automation for Men and Management*, Harper & Row (1965). The sentence first appeared in his 1960 chapter "The Corporation: Will It Be Managed by Machines?", in M. Anshen & G. L. Bach (eds.), *Management and Corporations 1985*, McGraw-Hill. The 1965 book reprints it. | Not achieved by 1985 |
| Within a generation the problem of creating AI would be substantially solved | Minsky 1967 | Not achieved |
| By about 2000, a machine would fool interrogators about 30% of the time in 5-minute tests | Turing 1950 | Roughly on target for narrow chat settings, though it says little about understanding (Weizenbaum's warning) |
| Micro-world successes would not scale without new ideas | Dreyfus 1965/72; Lighthill 1973 | Largely correct for symbolic AI of that era |

**Lesson:** early AI underestimated **common-sense knowledge, perception, scale and learning**. Those are exactly the areas that Turing's learning machine, Rosenblatt, Samuel, Selfridge and Dreyfus had emphasised.

---

## 11. Short answer: what 1950–1980 says about how to build intelligence and consciousness

1. **Intelligence** was approached through three broad strategies that were in tension and complemented each other:
   - (a) **Symbolic search over explicit knowledge**: Newell, Simon, McCarthy, Minsky, Feigenbaum.
   - (b) **Learning in adaptive networks and through experience**: Turing, Rosenblatt, Selfridge, Samuel, Holland.
   - (c) **Closed-loop, embodied, adaptive control**: Wiener, Ashby, Grey Walter.

   The founders agreed that **learning, search, representation, abstraction, self-improvement and handling complexity** were the essential ingredients (Dartmouth 1955; Minsky 1961). They disagreed about whether representations should be written down by hand or learned.
2. **Consciousness** was mostly **set aside as a scientific target** (Turing 1950). The constructive proposals were:
   - Minsky's **self-model** account (1965)
   - McCarthy's **engineered introspection** (1979)
   - **Functionalism** (Putnam)
   - Hofstadter's **strange loops** (1979)

   The main objections were Nagel's **subjective character** (1974) and Searle's **syntax is not semantics** (1980). By 1980 none of these positions offered an agreed engineering recipe for a conscious machine. The period ended with the question formulated more sharply, not answered.

---

## 12. Where to read the originals

- **Turing:** Turing Digital Archive, King's College Cambridge (`turingarchive.kings.cam.ac.uk`); Copeland, B. J. (ed.) (2004). *The Essential Turing*. Oxford UP.
- **MIT AI Lab memos** (Minsky AIM-306, Marr & Poggio AIM-357, and others): MIT DSpace, "AI Memos (1959–2004)" collection (`dspace.mit.edu`).
- **McCarthy's papers**, including "Programs with Common Sense", McCarthy & Hayes 1969, and "Ascribing Mental Qualities": his Stanford archive page (`jmc.stanford.edu`).
- **Dartmouth proposal:** *AI Magazine* 27(4) (2006), and the AAAI website.
- **RAND reports** (Dreyfus P-3244): `rand.org`, Papers series.
- **Lighthill report:** the Chilton Computing archive, which hosts the 1973 SRC paper symposium.
- **Anthologies:**
  - Feigenbaum & Feldman (eds.) (1963). *Computers and Thought*. McGraw-Hill.
  - Minsky (ed.) (1968). *Semantic Information Processing*. MIT Press.
  - Boden, M. A. (ed.) (1990). *The Philosophy of Artificial Intelligence*. Oxford UP. Reprints Turing 1950, McCulloch & Pitts, Searle 1980, Newell & Simon 1976 and others.
- **History:** Nilsson, N. J. (2010). *The Quest for Artificial Intelligence*. Cambridge UP; McCorduck, P. (1979/2004). *Machines Who Think*; Boden, M. A. (2006). *Mind as Machine*, 2 vols. Oxford UP.
