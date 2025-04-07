// Main JavaScript file for the Phishing Detection System

document.addEventListener('DOMContentLoaded', function() {
    console.log('Phishing Detection System initialized');
    
    // Initialize tooltips if Bootstrap is loaded
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
    
    // Handle file upload styling
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        const wrapper = input.closest('.file-upload-wrapper');
        if (wrapper) {
            const customUpload = wrapper.querySelector('.custom-file-upload');
            const fileNameDisplay = wrapper.querySelector('#file-upload-name');
            
            if (customUpload && fileNameDisplay) {
                input.addEventListener('change', function() {
                    if (this.files && this.files[0]) {
                        const fileName = this.files[0].name;
                        fileNameDisplay.textContent = fileName;
                        fileNameDisplay.classList.add('text-primary');
                        customUpload.classList.add('border-primary');
                    } else {
                        fileNameDisplay.textContent = 'No file selected';
                        fileNameDisplay.classList.remove('text-primary');
                        customUpload.classList.remove('border-primary');
                    }
                });
            }
        }
    });
    
    // Add visual feedback to form submission
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Processing...';
                submitBtn.disabled = true;
            }
        });
    });
    
    // Add copy to clipboard functionality for API keys or code examples
    const copyButtons = document.querySelectorAll('.copy-btn');
    copyButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            const textToCopy = this.getAttribute('data-copy');
            if (textToCopy) {
                navigator.clipboard.writeText(textToCopy).then(() => {
                    const originalText = this.innerHTML;
                    this.innerHTML = '<i class="fas fa-check me-1"></i> Copied!';
                    setTimeout(() => {
                        this.innerHTML = originalText;
                    }, 2000);
                }).catch(err => {
                    console.error('Failed to copy text: ', err);
                });
            }
        });
    });
    
    // Auto-dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
    alerts.forEach(alert => {
        setTimeout(() => {
            if (alert) {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            }
        }, 5000);
    });
    
    // Email analysis form validation enhancement
    const emailForm = document.getElementById('email-analysis-form');
    if (emailForm) {
        emailForm.addEventListener('submit', function(e) {
            const emailSubject = document.getElementById('email_subject');
            const emailContent = document.getElementById('email_content');
            
            if (emailSubject && emailContent) {
                if (emailSubject.value.trim() === '') {
                    alert('Please enter an email subject');
                    e.preventDefault();
                    return false;
                }
                
                if (emailContent.value.trim() === '') {
                    alert('Please enter email content');
                    e.preventDefault();
                    return false;
                }
            }
        });
    }
});
