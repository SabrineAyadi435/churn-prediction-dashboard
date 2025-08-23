document.addEventListener('DOMContentLoaded', function() {
    // File input display
    const fileInput = document.querySelector('input[type="file"]');
    if (fileInput) {
        const fileDisplay = document.querySelector('.file-upload p');
        fileInput.addEventListener('change', function(e) {
            fileDisplay.textContent = e.target.files[0]?.name || 'No file selected';
        });
    }

    // Initialize all Plotly charts
    document.querySelectorAll('.js-plotly-plot').forEach(plotEl => {
        const plotData = JSON.parse(plotEl.dataset.plot);
        Plotly.newPlot(plotEl, plotData.data, plotData.layout);
    });

    // Tooltip initialization
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Risk level badges
    document.querySelectorAll('.risk-badge').forEach(badge => {
        const riskLevel = badge.dataset.risk;
        if (riskLevel === 'high') {
            badge.classList.add('bg-danger');
        } else if (riskLevel === 'medium') {
            badge.classList.add('bg-warning', 'text-dark');
        } else {
            badge.classList.add('bg-success');
        }
    });
});

// Function to export data
function exportData(format) {
    alert(`Exporting data as ${format}...`);
    // In a real app, this would make an API call to export the data
}
// Upload Page Specific Interactions
if (document.getElementById('dropZone')) {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const uploadBtn = document.getElementById('uploadBtn');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const filePreview = document.getElementById('filePreview');
    const uploadProgress = document.getElementById('uploadProgress');

    // File Selection Handler
    browseBtn.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', handleFiles);
    
    // Drag and Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(event => {
        dropZone.addEventListener(event, preventDefaults, false);
    });
    
    ['dragenter', 'dragover'].forEach(event => {
        dropZone.addEventListener(event, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(event => {
        dropZone.addEventListener(event, unhighlight, false);
    });
    
    dropZone.addEventListener('drop', handleDrop, false);
    
    // Functions
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function highlight() {
        dropZone.classList.add('highlight');
    }
    
    function unhighlight() {
        dropZone.classList.remove('highlight');
    }
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
        handleFiles({target: fileInput});
    }
    
    function handleFiles(e) {
        const files = e.target.files;
        if (files.length) {
            const file = files[0];
            
            // Update UI
            fileName.textContent = file.name;
            fileSize.textContent = formatFileSize(file.size);
            filePreview.classList.add('has-file');
            uploadBtn.disabled = false;
        }
    }
    
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Accordion functionality
    const accordionHeader = document.querySelector('.accordion-header');
    if (accordionHeader) {
        accordionHeader.addEventListener('click', () => {
            const content = document.querySelector('.accordion-content');
            content.style.maxHeight = content.style.maxHeight ? null : content.scrollHeight + 'px';
            accordionHeader.querySelector('i').classList.toggle('fa-chevron-down');
            accordionHeader.querySelector('i').classList.toggle('fa-chevron-up');
        });
    }
}
// Add this to your main.js
document.addEventListener('DOMContentLoaded', function() {
    // Smooth scroll behavior for the entire app
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Custom scroll effects for main content
    const mainContent = document.querySelector('.app-main');
    if (mainContent) {
        let lastScrollTop = 0;
        
        mainContent.addEventListener('scroll', function() {
            const scrollTop = this.scrollTop;
            const scrollDirection = scrollTop > lastScrollTop ? 'down' : 'up';
            lastScrollTop = scrollTop <= 0 ? 0 : scrollTop;
            
            // You can add additional effects based on scroll direction
            // For example, hide/show header or other elements
        });
    }

    // Fade-in effect for elements when scrolling
    const fadeElements = document.querySelectorAll('.fade-on-scroll');
    if (fadeElements.length > 0) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in');
                }
            });
        }, { threshold: 0.1 });

        fadeElements.forEach(el => observer.observe(el));
    }
});