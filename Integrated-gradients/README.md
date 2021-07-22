# Integrated gradients implementation
## Introduction

## Description for each APIs and files
<p>To run the integrated gradient experimennt on the produced model, the main file, "Integrated_gradients_algorithm.py" file was used. I have implemented various self-made APIs, and in this section I will provide descriptions for each.</p>
<ul>
  <li>Integrated_gradients_algorithm.py
    <ul>
      <li>This python file runs integrated gradients on the saved model</li>
    </ul>
  </li>
  <li>Integrated_gradients_function.py
    <ul>
      <li>It is an API which provides the functions need to run the integrated gradeints algorithm</li>
    </ul>
  </li>
  <li>InputReader.py
    <ul>
      <li>This API provides functions that converts fasta files to one-hot encoding vector sequences to feed the network</li>
    </ul>
  </li>
  <li>Metrics.py
    <ul>
      <li>This API provides customized metrics such as f1 scores.</li>
    </ul>
  </li>
  <li>TestDataset2.fn
    <ul>
      <li>It is a test dataset file in fasta format</li>
    </ul>
  </li>
