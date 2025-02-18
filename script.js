let originalColumns = [];
let suggestedColumns = {};

async function getSuggestedColumnName(columnName) {
    try {
        const response = await fetch('http://localhost:8000/suggest-column', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                column_name: columnName
            })
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || 'API request failed');
        }

        return data;
    } catch (error) {
        console.error('Error:', error);
        alert(`Error: ${error.message || 'Failed to get column suggestion.'}`);
        return null;
    }
}

document.getElementById('excelFile').addEventListener('change', async function(e) {
    const file = e.target.files[0];
    const reader = new FileReader();

    reader.onload = async function(e) {
        const data = new Uint8Array(e.target.result);
        const workbook = XLSX.read(data, {type: 'array'});
        
        // Get the first sheet
        const firstSheet = workbook.Sheets[workbook.SheetNames[0]];
        
        // Get the range of the sheet
        const range = XLSX.utils.decode_range(firstSheet['!ref']);
        
        // Get column headers (assuming first row contains headers)
        const columns = [];
        for(let C = range.s.c; C <= range.e.c; ++C) {
            const cell = firstSheet[XLSX.utils.encode_cell({r: 0, c: C})];
            columns.push(cell ? cell.v : '');
        }

        originalColumns = columns;
        await displayColumns(columns);
    };

    reader.readAsArrayBuffer(file);
});

async function displayColumns(columns) {
    const columnList = document.getElementById('columnMapping');
    columnList.innerHTML = '';
    
    for (let i = 0; i < columns.length; i++) {
        const column = columns[i];
        const result = await getSuggestedColumnName(column);
        suggestedColumns[column] = result.suggested_name;

        const item = document.createElement('div');
        item.className = 'list-group-item';
        
        const sourceClass = result.source === 'excel_reference' ? 'bg-success text-white' : 
                          result.source === 'pdf_reference' ? 'bg-primary text-white' : 'bg-info text-white';
        
        const confidenceScore = result.confidence || 0;
        const confidenceClass = confidenceScore >= 0.8 ? 'bg-success' :
                              confidenceScore >= 0.5 ? 'bg-warning' : 'bg-danger';
        
        item.innerHTML = `
            <div class="mapping-row">
                <div class="col-12 col-md-4 mb-2 mb-md-0">
                    <span class="column-name" title="${column}">
                        ${column}
                    </span>
                </div>
                <div class="col-12 col-md-8">
                    <div class="input-group">
                        <span class="input-group-text">Suggested</span>
                        <input type="text" class="form-control suggested-name" 
                               value="${result.suggested_name || ''}" 
                               data-original="${column}">
                        <span class="input-group-text source-badge ${sourceClass}">
                            ${result.source || 'unknown'}
                        </span>
                        <span class="input-group-text">
                            <div class="confidence-badge ${confidenceClass}" 
                                 style="padding: 2px 6px; border-radius: 3px; color: white;">
                                ${(confidenceScore * 100).toFixed(0)}%
                            </div>
                        </span>
                    </div>
                </div>
            </div>
        `;
        columnList.appendChild(item);
    }
    
    document.getElementById('applyMappings').style.display = 'block';
}

document.getElementById('applyMappings').addEventListener('click', function() {
    const mappings = {};
    document.querySelectorAll('.suggested-name').forEach(input => {
        mappings[input.dataset.original] = input.value;
    });
    console.log('Final column mappings:', mappings);
    // Here you can implement the logic to apply these mappings to your Excel file
});
