def get_prompt_template(doc_type, extracted_text):
    if doc_type == "invoice":
        return f"""
You are analyzing an invoice document. Please extract:
- Invoice number
- Date
- Billed to (client)
- Items (description, quantity, unit price, total)
- Total amount
- Tax
- Due date

Document content:
{extracted_text[:4000]}
        """

    elif doc_type == "prescription":
        return f"""
This is a medical prescription. Please extract:
- Patient name (if present)
- Doctor name
- Medicines (name, dosage, frequency, duration)
- Additional instructions
-Tests

Prescription text:
{extracted_text[:4000]}
        """

    elif doc_type == "logistics":
        return f"""
Analyze this logistics receipt. Extract:
- Tracking ID
- Shipment date
- Sender and recipient
- Items and quantities
- Delivery address
- Status

Text:
{extracted_text[:4000]}
        """

    else:
        return f"""
Summarize the following document and extract any key insights or structured data:

{extracted_text[:4000]}
        """
