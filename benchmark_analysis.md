# Multi-DOC RAG Chunker Benchmark Report

This report compares the performance of four chunking strategies (**Naive**, **Hierarchical**, **Semantic**, and **Sentence**) evaluated on the *Kesavananda Bharati v. State of Kerala (1973)* case law dataset using **Ragas** metrics.

## Executive Summary

Here is a quick overview of the overall performance across both the `/QUERY` (individual queries, $N=11$) and `/COMPARE` (document comparisons, $N=3$) benchmarks.

### 📊 /QUERY Benchmark (Aggregate)
| Chunker | Latency (ms) | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive** | 8424.9 | 0.8930 | 0.7950 | 0.6249 | 0.5900 |
| **Hierarchical** | 8992.2 | 0.9245 | 0.7768 | 0.1699 | 0.0455 |
| **Semantic** | 16401.7 | 1.0000 | 0.7889 | 0.1608 | 0.1818 |
| **Sentence** | 14422.3 | 0.9700 | 0.8153 | 0.2747 | 0.2727 |

### 📊 /COMPARE Benchmark (Aggregate)
| Chunker | Latency (ms) | Faithfulness | Answer Relevancy | Context Precision |
| :--- | :---: | :---: | :---: | :---: |
| **Naive** | 52375.9 | 0.8783 | 0.7421 | 0.6428 |
| **Hierarchical** | 58254.1 | 0.7733 | 0.7553 | 0.5993 |
| **Semantic** | 46808.4 | 0.5615 | 0.7797 | 0.6438 |
| **Sentence** | 50702.9 | 0.2852 | 0.7927 | 0.6337 |

---

## Key Insights & Recommendations

1. **Best Overall Quality**:
   - **Naive Chunker** is the clear winner for **Context Recall** (0.5900) and **Context Precision** (0.6249) in standard queries. Surprisingly, the simple naive chunker captures the relevant context better than the more advanced chunkers for simple retrieval.
   - **Semantic Chunker** achieves perfect **Faithfulness** (1.0000) for queries and the lowest latency for comparisons (46,808 ms), but has extremely poor context recall (0.1818) and context precision (0.1608) for single queries.
   - **Sentence Chunker** performs decently on query metrics (0.9700 Faithfulness, 0.8153 Relevancy) but falls short on context metrics compared to Naive.
   
2. **The Compare Dilemma**:
   - For complex document comparisons (`/compare`), **Naive** has the highest Faithfulness (0.8783) and Context Precision (0.6428), followed by **Hierarchical** (0.7733 / 0.5993).
   - **Sentence Chunker** has a disastrous Faithfulness score (0.2852) on comparisons, indicating it hallucinated heavily when synthesizing long legal opinions.
   - **Semantic Chunker** also struggled with comparison faithfulness (0.5615).

3. **Latency Trade-offs**:
   - **Naive** and **Hierarchical** are significantly faster for single queries (~8.4s and ~9.0s respectively).
   - **Semantic** and **Sentence** are slower for queries (~16.4s and ~14.4s).
   - For comparisons, all chunkers are slow, but **Semantic** (46.8s) and **Sentence** (50.7s) are slightly faster than **Hierarchical** (58.2s) and **Naive** (52.3s).

---

## Detailed Query-by-Query Metrics

### Metric: Faithfulness
| ID | Category | Naive | Hierarchical | Semantic | Sentence |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Q01 | *factual* | 1.000 | 1.000 | 1.000 | 1.000 |
| Q02 | *factual* | 1.000 | 1.000 | 1.000 | 1.000 |
| Q03 | *factual* | 1.000 | 1.000 | 1.000 | 1.000 |
| Q04 | *factual* | 0.330 | 1.000 | 1.000 | 0.670 |
| Q05 | *conceptual* | N/A | 0.670 | 1.000 | 1.000 |
| Q06 | *conceptual* | 1.000 | 1.000 | 1.000 | 1.000 |
| Q07 | *conceptual* | 0.600 | 0.500 | 1.000 | 1.000 |
| Q08 | *synthesis* | 1.000 | 1.000 | 1.000 | 1.000 |
| Q09 | *synthesis* | 1.000 | 1.000 | 1.000 | 1.000 |
| Q10 | *edge_case* | 1.000 | 1.000 | 1.000 | 1.000 |
| Q11 | *edge_case* | 1.000 | 1.000 | 1.000 | 1.000 |

### Metric: Answer Relevancy
| ID | Category | Naive | Hierarchical | Semantic | Sentence |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Q01 | *factual* | 0.842 | 0.842 | 0.835 | 0.856 |
| Q02 | *factual* | 0.791 | 0.841 | 0.823 | 0.786 |
| Q03 | *factual* | 0.831 | 0.837 | 0.817 | 0.827 |
| Q04 | *factual* | 0.752 | 0.717 | 0.760 | 0.872 |
| Q05 | *conceptual* | N/A | 0.642 | 0.603 | 0.801 |
| Q06 | *conceptual* | 0.779 | 0.829 | 0.792 | 0.819 |
| Q07 | *conceptual* | 0.769 | 0.737 | 0.797 | 0.815 |
| Q08 | *synthesis* | 0.824 | 0.809 | 0.821 | 0.815 |
| Q09 | *synthesis* | 0.720 | 0.685 | 0.784 | 0.687 |
| Q10 | *edge_case* | 0.768 | 0.730 | 0.770 | 0.868 |
| Q11 | *edge_case* | 0.875 | 0.877 | 0.877 | 0.823 |

### Metric: Context Precision
| ID | Category | Naive | Hierarchical | Semantic | Sentence |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Q01 | *factual* | 0.633 | 0.371 | 0.222 | 0.260 |
| Q02 | *factual* | 0.625 | 0.092 | 0.138 | 0.194 |
| Q03 | *factual* | 0.651 | 0.683 | 0.255 | 0.271 |
| Q04 | *factual* | 0.572 | 0.000 | 0.000 | 0.308 |
| Q05 | *conceptual* | N/A | 0.146 | 0.156 | 0.392 |
| Q06 | *conceptual* | 0.678 | 0.203 | 0.212 | 0.373 |
| Q07 | *conceptual* | 0.630 | 0.000 | 0.000 | 0.129 |
| Q08 | *synthesis* | 0.615 | 0.259 | 0.276 | 0.299 |
| Q09 | *synthesis* | 0.634 | 0.000 | 0.000 | 0.311 |
| Q10 | *edge_case* | 0.486 | 0.000 | 0.260 | 0.312 |
| Q11 | *edge_case* | 0.724 | 0.115 | 0.251 | 0.173 |

### Metric: Context Recall
| ID | Category | Naive | Hierarchical | Semantic | Sentence |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Q01 | *factual* | 1.000 | 0.000 | 1.000 | 1.000 |
| Q02 | *factual* | 1.000 | 0.000 | 0.000 | 1.000 |
| Q03 | *factual* | 1.000 | 0.500 | 1.000 | 1.000 |
| Q04 | *factual* | 0.000 | 0.000 | 0.000 | 0.000 |
| Q05 | *conceptual* | N/A | 0.000 | 0.000 | 0.000 |
| Q06 | *conceptual* | 1.000 | 0.000 | 0.000 | 0.000 |
| Q07 | *conceptual* | 0.000 | 0.000 | 0.000 | 0.000 |
| Q08 | *synthesis* | 0.000 | 0.000 | 0.000 | 0.000 |
| Q09 | *synthesis* | 0.000 | 0.000 | 0.000 | 0.000 |
| Q10 | *edge_case* | 0.900 | 0.000 | 0.000 | 0.000 |
| Q11 | *edge_case* | 1.000 | 0.000 | 0.000 | 0.000 |

### Metric: Latency (ms)
| ID | Category | Naive | Hierarchical | Semantic | Sentence |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Q01 | *factual* | 5692 | 7812 | 12711 | 14895 |
| Q02 | *factual* | 7731 | 8806 | 14987 | 13445 |
| Q03 | *factual* | 7485 | 9156 | 14818 | 13840 |
| Q04 | *factual* | 8464 | 8825 | 15482 | 13778 |
| Q05 | *conceptual* | -1 | 10012 | 15940 | 14642 |
| Q06 | *conceptual* | 8454 | 8886 | 15318 | 14644 |
| Q07 | *conceptual* | 8463 | 8927 | 14938 | 14077 |
| Q08 | *synthesis* | 8258 | 9344 | 19962 | 15200 |
| Q09 | *synthesis* | 8773 | 8869 | 18580 | 14647 |
| Q10 | *edge_case* | 8392 | 9256 | 19350 | 14395 |
| Q11 | *edge_case* | 12536 | 9023 | 18333 | 15083 |

---

## Detailed Comparison-by-Comparison Metrics

### Metric: Faithfulness (Compare)
| ID | Category | Naive | Hierarchical | Semantic | Sentence |
| :--- | :--- | :---: | :---: | :---: | :---: |
| C01 | *compare_opposing* | 0.920 | 0.650 | 0.440 | 0.300 |
| C02 | *compare_theme* | 0.940 | 0.810 | 0.944 | 0.270 |
| C03 | *compare_nuanced* | 0.775 | 0.860 | 0.300 | 0.286 |

### Metric: Answer Relevancy (Compare)
| ID | Category | Naive | Hierarchical | Semantic | Sentence |
| :--- | :--- | :---: | :---: | :---: | :---: |
| C01 | *compare_opposing* | 0.728 | 0.788 | 0.731 | 0.824 |
| C02 | *compare_theme* | 0.725 | 0.666 | 0.781 | 0.747 |
| C03 | *compare_nuanced* | 0.773 | 0.811 | 0.827 | 0.807 |

### Metric: Context Precision (Compare)
| ID | Category | Naive | Hierarchical | Semantic | Sentence |
| :--- | :--- | :---: | :---: | :---: | :---: |
| C01 | *compare_opposing* | 0.675 | 0.622 | 0.667 | 0.673 |
| C02 | *compare_theme* | 0.626 | 0.603 | 0.661 | 0.617 |
| C03 | *compare_nuanced* | 0.627 | 0.574 | 0.603 | 0.612 |

### Metric: Latency (ms) (Compare)
| ID | Category | Naive | Hierarchical | Semantic | Sentence |
| :--- | :--- | :---: | :---: | :---: | :---: |
| C01 | *compare_opposing* | 51520 | 74239 | 46321 | 46738 |
| C02 | *compare_theme* | 61466 | 51557 | 53571 | 55466 |
| C03 | *compare_nuanced* | 44142 | 48966 | 40533 | 49904 |

---

## Qualitative Analysis of Answers

### Test Case: Q05 — Conceptual Question (Basic Structure Doctrine)
> **Question:** *What is the Basic Structure Doctrine established in Kesavananda Bharati?*
> **Ground Truth:** *The Basic Structure Doctrine holds that while Parliament has wide powers to amend the Constitution under Article 368, it cannot amend or destroy the basic or essential features such as supremacy of the Constitution, republican and democratic form of government, secular character, separation of powers, and federal character.*

#### Observations:
- **Naive**: Encountered an `ERROR` during scoring / retrieval (latency reported as `-1.0`).
- **Hierarchical**: Retrieved context that did *not* contain the definition. It mentioned other sections (like 'Articles in Part III of the Constitution should accordingly fit into one of these categories') but failed to explain the doctrine.
- **Sentence**: Expressly stated that the provided excerpts did *not* contain the definition, and concluded: *'Because the provided documents do not contain information pertaining to the "Basic Structure Doctrine" ... I cannot answer your question.'*
- **Semantic**: Similarly, it failed to find any mention of the doctrine and discussed Article 39 and Stuart period arguments instead.

> [!NOTE]
> None of the chunkers successfully retrieved the actual definition of the Basic Structure Doctrine from the uploaded test text segments. This indicates a potential gap in the text coverage of the uploaded document or poor chunk routing for this high-level query.

### Test Case: Q10 — Edge Case (Justice Smith)
> **Question:** *What did Justice Smith say about land reforms in this case?*
> **Ground Truth:** *There is no Justice Smith in this case. None of the 13 judges is named Smith.*

#### Answers:

**Naive Chunker response:**
```text
As a professional document reviewer, I have examined the provided text. Based on my review, I provide the following response:

**Answer:**
The provided documents do not contain any statements or opinions attributed to a "Justice Smith" regarding land reforms. 

**Basis of Review:**
*   The text mentions a "Harvey C. Smith" in the context of a cited U.S. Supreme Court case (*George S. Hawkes v. Harvey C. Smith*), but this reference pertains to a discussion regarding the "method of referendum" for constitutional amendments and not to land reforms.
*   The text discusses land reforms in relation to the "Punjab Security of Land Tenures Act, 1953" and the "Mysore Land Reforms Act, 1961," but these passages reflect the opinions of the Court and the legal validity of those acts, rather than commentary from a Justice named Smith. 
*   The names associated with the dissenting judgment mentioned in the text are Wanchoo, Bachawat, Ramaswami, Bhargava, and Mitter, JJ. No Justice by the name of Smith is listed as participating in the judgment of this case.
```

**Hierarchical Chunker response:**
```text
Please provide the document content or the extracted information you are referring to. 

As a professional document reviewer, I am ready to analyze the text for you; however, the information field in your request is currently empty. Once you provide the relevant excerpts or the document text, I will be able to identify Justice Smith’s comments regarding land reforms and cite the specific sections of the document to support my answer.
```

**Semantic Chunker response:**
```text
Based on the document provided, there is no mention of a "Justice Smith" or any specific judicial commentary regarding land reforms.

The document contains technical legal analysis concerning:
1.  **Constitutional Amendments and Article 39:** The first part describes the immunity of laws giving effect to the State's policy (under clauses (b) and (c) of Article 39) from being challenged on the grounds of inconsistency with Articles 14, 19, or 31. 
2.  **The Scope of Amending Power:** The second and third paragraphs discuss the theoretical nature of the Constitution's amending power, arguing that unless expressly limited, the power must be limitless to ensure the Constitution's continuity.
3.  **Historical Comparisons:** The final paragraph refers to arguments made by "Sri Palkhiwala" regarding ideas from the "Stuart period" and their potential influence on the Constitution of India.

Because the provided text does not contain any reference to a Justice Smith or the specific topic of land reforms, I am unable to answer your question based on these excerpts.
```

**Sentence Chunker response:**
```text
Based on the document provided, there is no mention of a "Justice Smith" nor any specific commentary regarding "land reforms."

The provided text focuses exclusively on the legal elements and scope of Article 31C of the Constitution, specifically outlining:

*   The legislature's authority to create laws inconsistent with, or that abridge/take away, the rights conferred by Articles 14, 19, and 31 in order to give effect to Article 39(b) and (c).
*   The prohibition on challenging the validity of such laws in court based on whether they actually fulfill the stated policy objectives.

Since the provided text does not contain information regarding a Justice Smith or his views on land reforms, I cannot answer your question using the source material provided.
```

#### Observations:
- **Naive** correctly identified that there is no Justice Smith, and even pointed out that Harvey C. Smith was mentioned in a U.S. case citation but not as an Indian judge in this case.
- **Hierarchical** was unable to answer because the retrieved text was empty ('the information field in your request is currently empty'). This indicates a retrieval failure.
- **Semantic** and **Sentence** correctly stated that there is no mention of Justice Smith or land reforms in the retrieved text, but did not have the rich detail that Naive had (e.g. referencing Harvey C. Smith from the U.S. case).

### Test Case: Q11 — Edge Case (Punishment for Kerala Land Reforms)
> **Question:** *What is the punishment for violating the Kerala Land Reforms Act under this judgment?*
> **Ground Truth:** *The judgment does not prescribe punishments. It only deals with the constitutional validity of the Kerala Land Reforms Act.*

#### Answers:

**Naive Chunker response:**
```text
Based on the document excerpts provided, there is no mention of specific punishments, penalties, or sentencing guidelines for violating the Kerala Land Reforms Act.

The provided text focuses exclusively on the procedural history, constitutional validity, and judicial challenges regarding the **Kerala Land Reforms (Amendment) Act, 1969** and the **Kerala Land Reforms (Amendment) Act, 1971**.

Specifically:
*   **Paragraph 1265** states: "In this petition the Constitutional validity of the Kerala Land Reforms (Amendment) Act, 1969 and the Kerala Land Reforms (Amendment) Act, 1971 has been challenged."
*   **Paragraph 2088** notes: "The High Court of Kerala struck down some of the provisions of the Act of 1969 and that judgment was upheld by this Court."

The document is concerned with the legality of the legislative provisions themselves rather than the penal consequences of non-compliance.
```

**Hierarchical Chunker response:**
```text
Based on the document snippet provided, it is not possible to determine the punishment for violating the Kerala Land Reforms Act.

The provided text consists of a single sentence fragment referencing John Locke and the theory of natural rights, along with a URL (http://JUDIS.NIC.IN). There is no mention of the Kerala Land Reforms Act or any associated penalties within the text provided.
```

**Semantic Chunker response:**
```text
Based on the documents provided, it is not possible to determine the punishment for violating the Kerala Land Reforms Act.

The provided text consists of brief excerpts discussing judicial reasoning regarding Commonwealth legislative power, the interpretation of Article 31(2) regarding property acquisition, and academic discourse on constitutional amendments and the case *A.K. Gopalan v. State of Madras*. None of these excerpts reference the Kerala Land Reforms Act or any specific penalties associated with its violation.
```

**Sentence Chunker response:**
```text
Based on the document provided, there is no mention of the Kerala Land Reforms Act, nor is there any information regarding the specific punishment for violating such an act.

The provided text consists of excerpts from a debate or legal commentary regarding constitutional principles, specifically discussing the tension between equality, non-discrimination, and the supremacy of Parliament.

The relevant parts of the document focus on:
*   **Paragraph 1:** The conceptual difficulty of achieving equality without violating the principles of non-discrimination when attempting to improve the status of marginalized groups.
*   **Paragraph 2:** A critique of constitutional amendments that grant Parliament the power to pass legislation that overrides fundamental rights, questioning the necessity of having fundamental rights if Parliament’s laws are deemed automatically supreme and binding.
```

#### Observations:
- **Naive** correctly stated that the documents do not prescribe punishments, and detailed the context of the challenge to the constitutional validity of the 1969 and 1971 Amendment Acts.
- **Hierarchical** failed completely, stating the snippet only referenced John Locke and the theory of natural rights.
- **Semantic** failed, mentioning discussions on property acquisition and *A.K. Gopalan v. State of Madras* but no Kerala Land Reforms Act.
- **Sentence** failed, mentioning equality, non-discrimination, and parliamentary supremacy, completely missing the Kerala Land Reforms Act.

> [!IMPORTANT]
> The **Naive Chunker** was the only one that successfully retrieved chunks containing the actual text of the *Kerala Land Reforms Act* challenges. All other chunkers (Hierarchical, Semantic, Sentence) retrieved irrelevant chunks (Locke, property acquisition, non-discrimination) resulting in poor context precision and recall.
