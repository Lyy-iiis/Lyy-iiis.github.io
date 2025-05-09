document.addEventListener('DOMContentLoaded', function() {
  // Button to show/hide additional demos
  const showMoreButton = document.getElementById('show-more-demos');
  const additionalDemos = document.getElementById('additional-demos');
  
  // 添加调试信息
  console.log('DOM loaded, initializing button functionality');
  console.log('Button found:', !!showMoreButton);
  console.log('Additional demos section found:', !!additionalDemos);
  
  if (showMoreButton && additionalDemos) {
    // 确保初始状态正确
    additionalDemos.style.display = 'none';
    
    // 设置初始按钮文本
    const buttonText = showMoreButton.querySelector('span:first-child');
    if (buttonText) {
      buttonText.innerHTML = '<strong>Show More Demos</strong>';
    }
    
    showMoreButton.addEventListener('click', function(e) {
      e.preventDefault(); // 防止默认行为
      console.log('Button clicked');
      
      // 使用计算样式而不是内联样式来检查可见性
      const computedStyle = window.getComputedStyle(additionalDemos);
      const isHidden = computedStyle.display === 'none';
      console.log('Is hidden:', isHidden);
      
      // Toggle visibility
      additionalDemos.style.display = isHidden ? 'flex' : 'none';
      
      // Update button text and icon
      const buttonText = showMoreButton.querySelector('span:first-child');
      const buttonIcon = showMoreButton.querySelector('.fas');
      
      if (buttonText && buttonIcon) {
        if (isHidden) {
          buttonText.innerHTML = '<strong>Hide Demos</strong>';
          buttonIcon.classList.remove('fa-chevron-down');
          buttonIcon.classList.add('fa-chevron-up');
        } else {
          buttonText.innerHTML = '<strong>Show More Demos</strong>';
          buttonIcon.classList.remove('fa-chevron-up');
          buttonIcon.classList.add('fa-chevron-down');
        }
      }
    });
    
    console.log('Button event listener attached');
  } else {
    console.error('Button or additional demos section not found');
  }
  
  // 添加备用初始化方法，以防DOMContentLoaded已经触发
  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    console.log('Document already loaded, checking elements again');
    setTimeout(function() {
      const showMoreButton = document.getElementById('show-more-demos');
      const additionalDemos = document.getElementById('additional-demos');
      
      if (showMoreButton && additionalDemos && !showMoreButton.hasAttribute('data-initialized')) {
        console.log('Initializing button after delay');
        showMoreButton.setAttribute('data-initialized', 'true');
        
        // 确保初始状态正确
        additionalDemos.style.display = 'none';
        
        showMoreButton.addEventListener('click', function(e) {
          e.preventDefault();
          const computedStyle = window.getComputedStyle(additionalDemos);
          const isHidden = computedStyle.display === 'none';
          
          additionalDemos.style.display = isHidden ? 'flex' : 'none';
          
          const buttonText = showMoreButton.querySelector('span:first-child');
          const buttonIcon = showMoreButton.querySelector('.fas');
          
          if (buttonText && buttonIcon) {
            if (isHidden) {
              buttonText.innerHTML = '<strong>Hide Demos</strong>';
              buttonIcon.classList.remove('fa-chevron-down');
              buttonIcon.classList.add('fa-chevron-up');
            } else {
              buttonText.innerHTML = '<strong>Show More Demos</strong>';
              buttonIcon.classList.remove('fa-chevron-up');
              buttonIcon.classList.add('fa-chevron-down');
            }
          }
        });
      }
    }, 500);
  }
});
