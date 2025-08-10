
### Size analysis
```
(rag-env) PS C:\Users\helios.lyons\Documents\git\osparser> python scan_pdf_pages.py --dir sources
Scanned 85 PDFs under sources (85 readable, 0 with errors).

Smallest document(s):
- All indicator assessment\p00945_bh2_benthic_habitat_conceptual_approach_qsr2023.pdf: 10 pages
- All other assessments\impacts_of_covid_belgian_report.pdf: 10 pages
- All other assessments\p00844_assessment_validation_process_qsr2023.pdf: 10 pages

Largest document(s):
- All other assessments\sambr_scientific_report_2017_final.pdf: 200 pages
(rag-env) PS C:\Users\helios.lyons\Documents\git\osparser> 
```

### Possible values
low bound - min_floor = 12
coefficient k = 14
upper_cap = 200 (max num pages?)
rerank headrom - top_k_initial = 300 (provide the options for the max cap? we need to scale this too?)