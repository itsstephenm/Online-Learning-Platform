/**
 * AI Adoption Analytics Platform JavaScript
 * Handles AJAX functionality for file uploads, model training, and data visualization
 */

// Initialize CSRF token for AJAX requests
function getCsrfToken() {
    return document.querySelector('[name=csrfmiddlewaretoken]').value;
}

// File upload handling
function initFileUpload() {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const dropArea = document.getElementById('drop-area');
    const processingIndicator = document.getElementById('processingIndicator');
    
    if (!uploadForm || !fileInput) return;
    
    // Set up event listeners for file drag & drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    // Highlight drop area when dragging over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropArea.classList.add('highlight');
    }
    
    function unhighlight() {
        dropArea.classList.remove('highlight');
    }
    
    // Handle dropped files
    dropArea.addEventListener('drop', function(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            fileInput.files = files;
            handleFileUpload(files[0]);
        }
    });
    
    // Handle file input change
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            handleFileUpload(this.files[0]);
        }
    });
    
    // Process the uploaded file
    function handleFileUpload(file) {
        // Check if file is CSV
        if (!file.name.endsWith('.csv')) {
            alert('Please upload a CSV file');
            return;
        }
        
        // Display file info
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileSize').textContent = formatFileSize(file.size);
        document.getElementById('filePreview').style.display = 'block';
        
        // Show processing indicator
        processingIndicator.style.display = 'flex';
        
        // Create FormData for AJAX upload
        const formData = new FormData();
        formData.append('csv_file', file);
        formData.append('csrfmiddlewaretoken', getCsrfToken());
        
        // Send AJAX request
        fetch('/quiz/api/upload-csv/', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Hide processing indicator
            processingIndicator.style.display = 'none';
            
            if (data.success) {
                // Show data preview
                updateDataPreview(data.preview_data);
                
                // Show success messages
                document.getElementById('successMessage').style.display = 'block';
                document.getElementById('successText').textContent = 
                    `Data successfully uploaded! Added ${data.added_records} new records.`;
                
                // Show skipped message if any
                if (data.skipped_records > 0) {
                    document.getElementById('skippedMessage').style.display = 'block';
                    document.getElementById('skippedText').textContent = 
                        `Skipped ${data.skipped_records} existing record${data.skipped_records > 1 ? 's' : ''} to avoid duplicates.`;
                }
                
                // Update stats
                updateStats(data.stats);
            } else {
                alert('Error: ' + (data.error || 'Failed to upload file'));
            }
        })
        .catch(error => {
            processingIndicator.style.display = 'none';
            alert('Error: ' + error);
        });
    }
    
    // Format file size for display
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }
    
    // Update data preview table
    function updateDataPreview(previewData) {
        const dataPreviewSection = document.getElementById('dataPreviewSection');
        const headerRow = document.getElementById('previewTableHeader');
        const tableBody = document.getElementById('previewTableBody');
        
        // Show the section
        dataPreviewSection.style.display = 'block';
        
        // Clear existing content
        headerRow.innerHTML = '';
        tableBody.innerHTML = '';
        
        // Add headers
        previewData.headers.forEach((header, index) => {
            const th = document.createElement('th');
            th.textContent = `${index + 1}. ${header}`;
            headerRow.appendChild(th);
        });
        
        // Add rows
        previewData.rows.forEach((rowData, rowIndex) => {
            const tr = document.createElement('tr');
            
            // Add row index cell
            const indexCell = document.createElement('td');
            indexCell.textContent = rowIndex;
            tr.appendChild(indexCell);
            
            // Add data cells
            rowData.forEach(cellData => {
                const td = document.createElement('td');
                td.textContent = cellData;
                tr.appendChild(td);
            });
            
            tableBody.appendChild(tr);
        });
    }
    
    // Update statistics display
    function updateStats(stats) {
        const statsSection = document.getElementById('statsSection');
        
        // Show the section
        statsSection.style.display = 'block';
        
        // Update values
        document.getElementById('totalResponses').textContent = stats.total_responses;
        document.getElementById('toolsCount').textContent = stats.tools_count;
        document.getElementById('avgUsage').textContent = stats.avg_usage;
    }
}

// Model training handling
function initModelTraining() {
    const trainModelBtn = document.getElementById('trainModelBtn');
    const processingIndicator = document.getElementById('processingIndicator');
    
    if (!trainModelBtn) return;
    
    trainModelBtn.addEventListener('click', function() {
        // Show processing indicator
        processingIndicator.style.display = 'flex';
        document.getElementById('processingText').textContent = 'Training model...';
        
        // Send AJAX request
        fetch('/quiz/api/train-model/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCsrfToken()
            },
            body: JSON.stringify({})
        })
        .then(response => response.json())
        .then(data => {
            // Hide processing indicator
            processingIndicator.style.display = 'none';
            
            if (data.success) {
                // Show success message
                document.getElementById('modelSuccessMessage').style.display = 'block';
                document.getElementById('modelSuccessText').textContent = 
                    `Model trained successfully! Accuracy: ${data.accuracy}`;
            } else {
                alert('Error: ' + (data.error || 'Failed to train model'));
            }
        })
        .catch(error => {
            processingIndicator.style.display = 'none';
            alert('Error: ' + error);
        });
    });
}

// Initialize dashboard functionality
document.addEventListener('DOMContentLoaded', function() {
    // Initialize file upload
    initFileUpload();
    
    // Initialize model training
    initModelTraining();
}); 