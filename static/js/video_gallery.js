document.addEventListener('DOMContentLoaded', () => {
    const videoPreviewContainers = document.querySelectorAll('.video-preview-container');
  
    videoPreviewContainers.forEach(container => {
      container.addEventListener('mouseenter', () => {
        const previewImg = container.querySelector('.video-preview');
        const video = container.querySelector('.video-content');
        previewImg.style.display = 'none';
        video.style.display = 'block';
        video.play();
      });
  
      container.addEventListener('mouseleave', () => {
        const video = container.querySelector('.video-content');
        const previewImg = container.querySelector('.video-preview');
        video.pause();
        video.currentTime = 0;
        video.style.display = 'none';
        previewImg.style.display = 'block';
      });
    });
  });