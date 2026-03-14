const modelViewerGallary = document.querySelector('#modelViewerGallary');
modelViewerGallary.isTextured = false;

// Initialize the selection panel images
$('#gallerySelectionPanel .selectable-image').each((i, img) => {
    img.src = `https://hi3dgen-release.tos-cn-beijing.volces.com/gallery/${img.getAttribute('name')}.png`;
})

// Add event listener to the selection panel
const gallerySelectionPanel = document.getElementById('gallerySelectionPanel');
gallerySelectionPanel.addEventListener('click', async function(event) {
    const img = event.target.closest('.selectable-image'); 
    
    if (!img || img.classList.contains('selected')) 
        return;
   
    gallerySelectionPanel.querySelectorAll('.selectable-image').forEach(function(image) {
        image.classList.remove('selected');
    });
    img.classList.add('selected');

    const name = img.getAttribute('name');

    modelViewerGallary.src = `https://hi3dgen-release.tos-cn-beijing.volces.com/gallery/${name}.glb`;
    modelViewerGallary.texturePath = `https://hi3dgen-release.tos-cn-beijing.volces.com/gallery/${name}_normal.png`;
    modelViewerGallary.resetView();
    modelViewerGallary.showPoster();
});

// // Set the toggle buttons
// toggleGalleryLeftButton = document.querySelector('#toggleTexturedGallery .toggle-left');
// toggleGalleryRightButton = document.querySelector('#toggleTexturedGallery .toggle-right');

// toggleGalleryLeftButton.addEventListener('click', function() {
//     toggleGalleryLeftButton.classList.add('active');
//     toggleGalleryRightButton.classList.remove('active');
//     modelViewerGallary.setTextured(false);
// });

// toggleGalleryRightButton.addEventListener('click', function() {
//     toggleGalleryRightButton.classList.add('active');
//     toggleGalleryLeftButton.classList.remove('active');
//     modelViewerGallary.setTextured(false);
// });

// Initialize the model viewer with selected model
$(document).ready(() => {
    const name = document.querySelector('#gallerySelectionPanel .selectable-image.selected').getAttribute('name');

    modelViewerGallary.src = `https://hi3dgen-release.tos-cn-beijing.volces.com/gallery/${name}.glb`;
    modelViewerGallary.texturePath = `https://hi3dgen-release.tos-cn-beijing.volces.com/gallery/${name}_normal.png`;
    
    modelViewerGallary.resetView();

    modelViewerGallary.showPoster();  
});
