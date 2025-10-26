# Drooms Help Center

The Drooms Redaction feature is a very convenient and time-saving way to protect sensitive information and prepare documents in compliance to company and legal privacy policies, like the European GPDR. Drooms allows not only to redact documents individually, but also multiple documents per batch.

---

## Overview of Batch Redaction (updated)

Redaction 1.1 provides significant updates for the redaction feature, including the ability to batch redact multiple documents and folders as well as general redaction improvements:

- Batch redact multiple documents and folders to apply search terms and AI categories.
- Selection redaction to create markups by clicking on content in the document viewer.
- Improved reporting on applied and activated markups.
- Dutch as 6th language to apply terms.
- Monetary amounts as 26th redaction category.

---

## When to Use Batch Redaction

Batch redacting is ideal for removing common, general terms and data categories that must be redacted across all documents. For specific, detailed information, use single document redaction to address each document individually.

---

## Batch Redaction Workflow

To initiate batch redaction, select at least two documents or at least one folder. It is possible to select documents/folders from different parts of the index and between different granularities.

Once the user selects the batch of documents to redact, a window with a pre-redaction report will inform which files can be redacted and which cannot. Files that cannot be redacted are files that have not been OCRed, contain too little text, or are already redacted copies.

Users can select the redaction terms and categories that are applied for all documents in the selected batch. Category selection works like regular redaction. For the search terms, the user may enter the terms by typing or copy and paste multiple terms at once, separating them by comma or space (e.g., copying from an Excel spreadsheet).

The user needs to define the target and permission for the redacted files. Once the user submits the request, the documents are processed and notifications inform the user about the current progress. Batch redaction tries to replicate the structure and order of the source documents. Clicking “View result” in the notification tab filters for all processed documents. These can then be further edited with single document redaction.

---

## Step-by-Step: How to Batch Redact Documents

### 1. Start Drooms and Sign In

Begin by starting Drooms and signing in to your account. Select the project, and if your project contains multiple data rooms, select the appropriate data room or asset containing the documents you want to redact.

---

### 2. Access the Data Room Index

Select 'Documents' in the project's tab bar to access the data room index. Here, you can choose either the folder or the specific documents you want to redact.

---

### 3. Select Documents or Folders for Batch Redaction

To initiate batch redaction, select at least two documents or at least one folder. You can select documents or folders from different parts of the index and at different levels of granularity.

![screenshot](ID:IMG006)

---

### 4. Open the Index Context Menu

Right-click on the folder or one of the documents you want to redact. This opens the Index context menu. Select the option 'Batch redact', if available. (Batch Redaction must be enabled for your project.)

What you see: The context menu will display the 'Batch redact' option among others.

![screenshot](ID:IMG003)

---

### 5. Review the Batch Redaction Modal Dialog

The 'Batch Redaction' modal dialog will open, listing the files selected for batch redaction. Files that cannot be redacted per batch (such as already redacted files) will be marked with a red cross on the right as not redactable.

What you see: A list of files with indicators showing which can and cannot be redacted.

![screenshot](ID:IMG004)

---

### 6. Enter Redaction Terms and Categories

Enter the terms and/or categories to be redacted. To enter a term and separate it from further terms, use a comma or the Enter key. Note: Specific document areas cannot be selected and batch redacted.

What you see: Fields to enter terms and select categories for redaction.

---

### 7. Start the Redaction Process

Click 'Redact' once you have entered all terms and selected all categories to be redacted.

---

### 8. Select the Destination for Redacted Documents

When batch redacting documents, you are prompted directly to select the destination where the redacted document versions should be saved.

You can save the redacted documents as new documents:
- 'in' a folder
- 'before' a specific document index point
- 'after' a specific document index point

What you see: A dialog to choose the save location and permission inheritance.

![screenshot](ID:IMG005)

Administrators must select whether group permissions should be inherited from the destination parent folder or not. For documents redacted by regular users, group permissions will always be inherited.

---

### 9. Save the Redacted Documents

Click 'Save' to save the redacted document versions at the desired positions in the index.

---

### 10. Further Individual Redaction (Optional)

You may now continue to redact the documents processed per batch in more detail individually, using single document redaction features.

---

## General Redaction Improvements

### Selection Redaction: Create Markups by Clicking Content

Previously, terms not found by AI categories needed to be searched for, which was inconvenient for longer terms. Users now have an efficient alternative by simply clicking on the content (terms or images) within the document viewer. This selection method also works for images (e.g., logos).

Note: Only content recognized by OCR can be redacted in this way.

![screenshot](ID:IMG007)
![screenshot](ID:IMG008)

---

### Improved Reporting on Applied and Activated Markups

When a user applied AI categories in version 1.0, there was no proper overview of the found and activated category types. Version 1.1 introduces a count of activated and found markups when clicking on the AI category icon again. For example, 19/19 indicates that 19 occurrences of “organization” were found in the document, and all 19 are activated. If the user unchecks two occurrences, the count tracks 17/19 organizations are now activated.

---

### Dutch as 6th Language for AI Redaction

The AI redaction feature now supports Dutch, allowing you to search for 26 different categories in Dutch documents.

![screenshot](ID:IMG010)

---

### Monetary Amounts as New Redaction Category

AI redaction can now identify monetary amounts such as prices and salaries.

---

## Batch Redaction Pricing and Availability

Batch Redaction is available in all product offerings that include single document redaction, such as TDRs, Flex datarooms with redaction enabled, and Lifecycle Premium datarooms. However, Batch Redaction is technically a separate module and could be priced separately in the future.

---

## Additional Resources

- Read more about redacting documents
- Read more about redacting search terms
- Read more about redacting terms matching sensitive data categories
- Read more about redacting selected document areas

---

## Tips and Guidelines

- Only documents that have not been redacted before can be redacted per batch.
- Batch redacting allows you to redact terms and data categories, but not to set areas for redacting across documents.
- Use batch redaction for common/general terms and categories, and single document redaction for specific details.

---

This concludes the updated guide for Batch Redacting Documents in Drooms.