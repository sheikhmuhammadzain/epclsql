/**
 * Response Formatter Module
 * Handles formatting of different types of AI responses with proper styling and structure
 */

class ResponseFormatter {
    constructor() {
        this.formatters = {
            'department_analysis': this.formatDepartmentAnalysis.bind(this),
            'findings_analysis': this.formatFindingsAnalysis.bind(this),
            'category_analysis': this.formatCategoryAnalysis.bind(this),
            'cost_analysis': this.formatCostAnalysis.bind(this),
            'general': this.formatGeneralResponse.bind(this)
        };
    }

    /**
     * Main formatting method - detects response type and applies appropriate formatter
     */
    formatResponse(result) {
        const responseType = this.detectResponseType(result);
        const formatter = this.formatters[responseType] || this.formatters['general'];
        return formatter(result);
    }

    /**
     * Detect the type of response based on content and SQL queries
     */
    detectResponseType(result) {
        const answer = result.response?.answer?.toLowerCase() || '';
        const sql = result.executed_sql?.[0]?.toLowerCase() || '';

        if (answer.includes('departments requiring') || answer.includes('incident analysis by department') || sql.includes('department')) {
            return 'department_analysis';
        }
        if (answer.includes('audit findings') || answer.includes('inspection findings') || sql.includes('findings')) {
            return 'findings_analysis';
        }
        if (answer.includes('incidents by category') || sql.includes('category')) {
            return 'category_analysis';
        }
        if (answer.includes('cost') || answer.includes('expensive') || sql.includes('total_cost')) {
            return 'cost_analysis';
        }
        return 'general';
    }

    /**
     * Format department analysis responses
     */
    formatDepartmentAnalysis(result) {
        let content = `<div class="response-container department-analysis">`;
        
        // Header
        content += `<div class="response-header">
            <h3><i class="fas fa-building"></i> Department Safety Analysis</h3>
        </div>`;

        // Main content
        const answer = result.response.answer;
        
        if (answer.includes('DEPARTMENTS REQUIRING IMMEDIATE ATTENTION')) {
            content += this.formatDepartmentPriorityList(answer);
        } else if (answer.includes('INCIDENT ANALYSIS BY DEPARTMENT')) {
            content += this.formatDepartmentIncidentList(answer);
        } else {
            content += `<div class="response-content">${this.formatText(answer)}</div>`;
        }

        // SQL Query section
        if (result.executed_sql && result.executed_sql.length > 0) {
            content += this.formatSQLSection(result.executed_sql);
        }

        // Recommendations
        if (result.response.recommendations && result.response.recommendations.length > 0) {
            content += this.formatRecommendations(result.response.recommendations);
        }

        // Metadata
        content += this.formatMetadata(result);
        content += `</div>`;

        return content;
    }

    /**
     * Format department priority list with visual indicators
     */
    formatDepartmentPriorityList(answer) {
        let content = `<div class="priority-analysis">`;
        
        const lines = answer.split('\n');
        let inDepartmentList = false;
        let inSummary = false;

        for (let line of lines) {
            line = line.trim();
            if (!line) continue;

            if (line.includes('DEPARTMENTS REQUIRING IMMEDIATE ATTENTION')) {
                content += `<div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle"></i> ${line.replace(/🚨\s*\*\*|\*\*/g, '')}
                </div>`;
                inDepartmentList = true;
            } else if (line.includes('Based on open incidents')) {
                content += `<p class="analysis-subtitle">${line}</p>`;
            } else if (line.match(/^\d+\.\s+\*\*.*\*\*/)) {
                // Department entry
                content += this.formatDepartmentEntry(line);
            } else if (line.includes('📈 **SUMMARY**')) {
                inSummary = true;
                content += `<div class="summary-section">
                    <h4><i class="fas fa-chart-line"></i> Summary</h4>
                    <p>${line.replace(/📈\s*\*\*SUMMARY\*\*:\s*/, '')}</p>
                </div>`;
            } else if (line.startsWith('   📊') || line.startsWith('   🚨') || line.startsWith('   ✅') || line.startsWith('   💰')) {
                // Department details
                content += this.formatDepartmentDetail(line);
            }
        }

        content += `</div>`;
        return content;
    }

    /**
     * Format individual department entry
     */
    formatDepartmentEntry(line) {
        const match = line.match(/^(\d+)\.\s+\*\*(.*?)\*\*\s+(🔴|🟡|🟢|⚪)\s+(.*)/);
        if (!match) return `<div class="department-entry">${line}</div>`;

        const [, number, department, priority, level] = match;
        const priorityClass = this.getPriorityClass(priority);
        
        return `<div class="department-entry ${priorityClass}">
            <div class="department-header">
                <span class="department-number">${number}</span>
                <span class="department-name">${department}</span>
                <span class="priority-badge ${priorityClass}">${priority} ${level}</span>
            </div>
            <div class="department-details">`;
    }

    /**
     * Format department detail line
     */
    formatDepartmentDetail(line) {
        const iconMap = {
            '📊': 'chart-bar',
            '🚨': 'exclamation-triangle',
            '✅': 'check-circle',
            '💰': 'dollar-sign'
        };

        for (let [emoji, icon] of Object.entries(iconMap)) {
            if (line.includes(emoji)) {
                const text = line.replace(/\s*📊|🚨|✅|💰\s*/, '').trim();
                return `<div class="detail-item">
                    <i class="fas fa-${icon}"></i>
                    <span>${text}</span>
                </div>`;
            }
        }
        
        return `<div class="detail-item">${line.trim()}</div></div>`;
    }

    /**
     * Format findings analysis responses
     */
    formatFindingsAnalysis(result) {
        let content = `<div class="response-container findings-analysis">`;
        
        content += `<div class="response-header">
            <h3><i class="fas fa-search"></i> Findings Analysis</h3>
        </div>`;

        const answer = result.response.answer;
        content += this.formatFindingsList(answer);

        if (result.executed_sql && result.executed_sql.length > 0) {
            content += this.formatSQLSection(result.executed_sql);
        }

        if (result.response.recommendations && result.response.recommendations.length > 0) {
            content += this.formatRecommendations(result.response.recommendations);
        }

        content += this.formatMetadata(result);
        content += `</div>`;

        return content;
    }

    /**
     * Format findings list
     */
    formatFindingsList(answer) {
        let content = `<div class="findings-list">`;
        
        const lines = answer.split('\n');
        let currentTitle = '';

        for (let line of lines) {
            line = line.trim();
            if (!line) continue;

            if (line.includes('findings:')) {
                currentTitle = line;
                content += `<h4>${line}</h4><ol class="findings-items">`;
            } else if (line.match(/^\d+\.\s+/)) {
                const [, number, finding, count] = line.match(/^(\d+)\.\s+(.*?):\s*(\d+)$/) || [, '', line, ''];
                content += `<li class="finding-item">
                    <div class="finding-text">${finding || line}</div>
                    ${count ? `<div class="finding-count">${count}</div>` : ''}
                </li>`;
            }
        }

        content += `</ol></div>`;
        return content;
    }

    /**
     * Format category analysis responses
     */
    formatCategoryAnalysis(result) {
        let content = `<div class="response-container category-analysis">`;
        
        content += `<div class="response-header">
            <h3><i class="fas fa-tags"></i> Category Analysis</h3>
        </div>`;

        content += `<div class="response-content">${this.formatText(result.response.answer)}</div>`;

        if (result.executed_sql && result.executed_sql.length > 0) {
            content += this.formatSQLSection(result.executed_sql);
        }

        if (result.response.recommendations && result.response.recommendations.length > 0) {
            content += this.formatRecommendations(result.response.recommendations);
        }

        content += this.formatMetadata(result);
        content += `</div>`;

        return content;
    }

    /**
     * Format cost analysis responses
     */
    formatCostAnalysis(result) {
        let content = `<div class="response-container cost-analysis">`;
        
        content += `<div class="response-header">
            <h3><i class="fas fa-dollar-sign"></i> Cost Analysis</h3>
        </div>`;

        content += `<div class="response-content">${this.formatText(result.response.answer)}</div>`;

        if (result.executed_sql && result.executed_sql.length > 0) {
            content += this.formatSQLSection(result.executed_sql);
        }

        if (result.response.recommendations && result.response.recommendations.length > 0) {
            content += this.formatRecommendations(result.response.recommendations);
        }

        content += this.formatMetadata(result);
        content += `</div>`;

        return content;
    }

    /**
     * Format general responses
     */
    formatGeneralResponse(result) {
        let content = `<div class="response-container general-response">`;
        
        content += `<div class="response-header">
            <h3><i class="fas fa-robot"></i> Assistant Response</h3>
        </div>`;

        content += `<div class="response-content">${this.formatText(result.response.answer)}</div>`;

        if (result.executed_sql && result.executed_sql.length > 0) {
            content += this.formatSQLSection(result.executed_sql);
        }

        if (result.response.recommendations && result.response.recommendations.length > 0) {
            content += this.formatRecommendations(result.response.recommendations);
        }

        content += this.formatMetadata(result);
        content += `</div>`;

        return content;
    }

    /**
     * Format SQL section
     */
    formatSQLSection(sqlQueries) {
        let content = `<div class="sql-section">
            <h4><i class="fas fa-database"></i> SQL Query</h4>`;
        
        sqlQueries.forEach(sql => {
            content += `<div class="sql-code">
                <pre><code>${this.escapeHtml(sql)}</code></pre>
                <button class="copy-sql-btn" onclick="copyToClipboard('${this.escapeHtml(sql).replace(/'/g, "\\'")}')">
                    <i class="fas fa-copy"></i> Copy
                </button>
            </div>`;
        });
        
        content += `</div>`;
        return content;
    }

    /**
     * Format recommendations section
     */
    formatRecommendations(recommendations) {
        let content = `<div class="recommendations-section">
            <h4><i class="fas fa-lightbulb"></i> Recommendations</h4>
            <div class="recommendations-list">`;

        let currentCategory = '';
        
        for (let rec of recommendations) {
            if (rec.includes('IMPLEMENTATION TIMELINE:')) {
                content += `<div class="timeline-section">
                    <h5><i class="fas fa-clock"></i> Implementation Timeline</h5>
                    <ul class="timeline-list">`;
            } else if (rec.startsWith('   •')) {
                content += `<li class="timeline-item">${rec.replace('   •', '').trim()}</li>`;
            } else if (rec.trim() === '') {
                // Skip empty lines
                continue;
            } else {
                if (currentCategory) {
                    content += `</ul></div>`;
                }
                content += `<div class="recommendation-item">
                    <div class="rec-text">${this.formatRecommendationText(rec)}</div>
                </div>`;
            }
        }

        content += `</div></div>`;
        return content;
    }

    /**
     * Format recommendation text with icons
     */
    formatRecommendationText(text) {
        const iconMap = {
            '🚨': 'exclamation-triangle',
            '📊': 'chart-bar',
            '🔍': 'search',
            '📚': 'book',
            '⚡': 'bolt',
            '📈': 'chart-line',
            '🎯': 'bullseye',
            '💰': 'dollar-sign',
            '⏰': 'clock',
            '📅': 'calendar',
            '👥': 'users',
            '🔄': 'sync',
            '🏢': 'building',
            '🏭': 'industry',
            '🔧': 'wrench',
            '👷': 'hard-hat',
            '📋': 'clipboard-list'
        };

        for (let [emoji, icon] of Object.entries(iconMap)) {
            if (text.includes(emoji)) {
                text = text.replace(emoji, `<i class="fas fa-${icon}"></i>`);
                break;
            }
        }

        return text;
    }

    /**
     * Format metadata section
     */
    formatMetadata(result) {
        return `<div class="response-metadata">
            <div class="metadata-item">
                <i class="fas fa-clock"></i>
                <span>Execution Time: ${result.execution_time?.toFixed(2) || 'N/A'}s</span>
            </div>
            <div class="metadata-item">
                <i class="fas fa-calendar"></i>
                <span>Timestamp: ${new Date(result.timestamp).toLocaleString()}</span>
            </div>
        </div>`;
    }

    /**
     * Get CSS class for priority level
     */
    getPriorityClass(priority) {
        const priorityMap = {
            '🔴': 'critical',
            '🟡': 'high',
            '🟢': 'medium',
            '⚪': 'low'
        };
        return priorityMap[priority] || 'low';
    }

    /**
     * Format text with basic markdown-like formatting
     */
    formatText(text) {
        if (!text) return '';
        
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>')
            .replace(/(\d+)\.\s+/g, '<br><strong>$1.</strong> ');
    }

    /**
     * Escape HTML characters
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Utility function for copying SQL to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        // Show temporary success message
        const btn = event.target.closest('.copy-sql-btn');
        const originalText = btn.innerHTML;
        btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
        btn.classList.add('copied');
        
        setTimeout(() => {
            btn.innerHTML = originalText;
            btn.classList.remove('copied');
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy text: ', err);
    });
}

// Export for use in main application
window.ResponseFormatter = ResponseFormatter;
