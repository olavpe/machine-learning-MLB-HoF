---
title: Baseball Hall of Fame Project
subtitle: TTK28 Modeling with Neural Networks Exam
author: Olav Landmark Pedersen
date: 26.11.2020
---



# Goal

\newcommand\Fontvi{\fontsize{8}{9.6}\selectfont}

Determine if an eligible Major League Baseball (MLB) player
will make it into the Baseball Hall of Fame (HoF) based on career statistics.

- BBWAA and Veterans Committee can nominate and elect individuals to HoF
- Subjective voting system
- MLB players are eligible after playing 10 years
- Player, managers, umpires and executives are eligible


# Dataset

- Sean Lahman's Baseball Archive[^1]

- Imbalanced dataset:
  - 226 players in HoF
  - 3190 eligible players not in HoF
  - 14:1 ratio of non-HoF to HoF

- Hold-out split: Training 80%, Testing 20%

- Difference in quality features by positions
  - Good for batters (OPS, SLG, OBS, etc.)
  - Okay for pitchers (WHIP, K/BB, K/9, etc.)
  - Poor for other defense (FFRA, UZR, etc.)

[^1]: http://www.seanlahman.com/baseball-archive/

# Model


\begin{columns}

\column{0.4\textwidth}
\Fontvi
\begin{itemize}
\item Depth: 3
\item Width:
\begin{itemize}
\Fontvi
\item 10
\item 5
\item 1
\end{itemize}
\item Activation:
\begin{itemize}
\Fontvi
\item ReLU
\item ReLU
\item Sigmoid
\end{itemize}
\item Loss: Binary Cross-entropy
\item Batch Size: 1
\item Epochs: 50 
\item Optimizer: Adam
\end{itemize}

\column{0.6\textwidth}
\includegraphics[width=\textwidth]{network.png}

\end{columns}

# Implementation

- Fair amount of pre-processing
- Primarily used Keras and Scikit-learn tools
- Stratified 10-fold Cross-validation
- Final Test only done at the end.

- Handling and small imbalanced dataset:
  - Tried undersampling 
  - Class Weights
  - Small Batch Size

# Results

\begin{columns}

\column{0.5\textwidth}
\begin{itemize}
\item Accuracy:  0.95
\item AUROC:     0.96
\item Precision: 0.66
\item Recall:    0.47
\item F1:        0.55
\end{itemize}
\includegraphics[width=1.1\textwidth]{final_confusion_mat.png}

\column{0.5\textwidth}
\includegraphics[width=1.1\textwidth]{final_ROC_curve.png}

\end{columns}

# 

<!-- \begin{frame}[plain, c] -->
\begin{center}
\Huge Thank you for your attention!
\end{center}
<!-- \end{frame} -->
