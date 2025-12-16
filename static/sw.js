// Nama cache
const CACHE_NAME = 'plant-ai-v1';

// Install event
self.addEventListener('install', (event) => {
    console.log('Service Worker: Installed');
});

// Fetch event (Agar bisa load gambar offline/cache)
self.addEventListener('fetch', (event) => {
    event.respondWith(
        fetch(event.request).catch(() => caches.match(event.request))
    );
});