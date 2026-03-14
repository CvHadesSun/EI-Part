const enlargedImagesContainer = document.querySelector('#enlarged-images-container');
const enlargedImage = document.querySelector('#enlarged-image');
const enlargedImageNormal = document.querySelector('#enlarged-image-normal');

document.querySelectorAll('.selectable-image').forEach(img => {
  img.addEventListener('mouseenter', function() {
    const imgRect = img.getBoundingClientRect();
    const src = img.src;

    let normalSrc;
    if (src.includes('_color')) {
      normalSrc = src.replace('_color', '_normal');
    } else {
      normalSrc = src.replace(/(\.[^/.]+)$/, '_normal$1');
    }

    enlargedImage.src = src;
    enlargedImageNormal.src = normalSrc;

    enlargedImagesContainer.style.opacity = 1;
    enlargedImagesContainer.style.left = Math.max(20, Math.min(window.innerWidth - enlargedImagesContainer.offsetWidth - 20, imgRect.left + imgRect.width / 2 - enlargedImagesContainer.offsetWidth / 2)) + 'px';
    enlargedImagesContainer.style.top = imgRect.top - 10 - enlargedImagesContainer.offsetHeight + 'px';
  });

  img.addEventListener('mouseleave', function() {
    enlargedImagesContainer.style.opacity = 0;
  });

  img.addEventListener('touchend', function() {
    enlargedImagesContainer.style.opacity = 0;
  });
});
