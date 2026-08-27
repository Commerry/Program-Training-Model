import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/stores/authStore'

const routes = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/LoginView.vue'),
    meta: { guestOnly: true, title: 'Sign in' }
  },
  {
    path: '/register',
    name: 'Register',
    component: () => import('@/views/RegisterView.vue'),
    meta: { guestOnly: true, title: 'Create account' }
  },

  { path: '/', redirect: { name: 'Dashboard' } },
  {
    path: '/dashboard',
    name: 'Dashboard',
    component: () => import('@/views/DashboardView.vue'),
    meta: { requiresAuth: true, title: 'Dashboard' }
  },
  {
    path: '/projects',
    name: 'Projects',
    component: () => import('@/views/ProjectsView.vue'),
    meta: { requiresAuth: true, title: 'Projects' }
  },
  {
    path: '/projects/:name',
    name: 'ProjectDetail',
    component: () => import('@/views/ProjectDetailView.vue'),
    props: true,
    meta: { requiresAuth: true, title: 'Project' }
  },
  {
    path: '/projects/:name/annotate/:filename',
    name: 'Annotate',
    component: () => import('@/views/AnnotateView.vue'),
    props: true,
    meta: { requiresAuth: true, title: 'Annotate' }
  },
  {
    path: '/projects/:name/train',
    name: 'Train',
    component: () => import('@/views/TrainView.vue'),
    props: true,
    meta: { requiresAuth: true, title: 'Train' }
  },
  {
    path: '/datasets',
    name: 'Datasets',
    component: () => import('@/views/DatasetsView.vue'),
    meta: { requiresAuth: true, title: 'Datasets' }
  },
  {
    path: '/models',
    name: 'Models',
    component: () => import('@/views/ModelsView.vue'),
    meta: { requiresAuth: true, title: 'Models' }
  },
  {
    path: '/test-model',
    name: 'TestModel',
    component: () => import('@/views/TestModelView.vue'),
    meta: { requiresAuth: true, title: 'Test model' }
  },
  {
    path: '/history',
    name: 'History',
    component: () => import('@/views/HistoryView.vue'),
    meta: { requiresAuth: true, title: 'Training history' }
  },
  {
    path: '/analytics',
    name: 'Analytics',
    component: () => import('@/views/AnalyticsView.vue'),
    meta: { requiresAuth: true, title: 'Analytics' }
  },
  {
    path: '/settings',
    name: 'Settings',
    component: () => import('@/views/SettingsView.vue'),
    meta: { requiresAuth: true, title: 'Settings' }
  },

  {
    path: '/:pathMatch(.*)*',
    name: 'NotFound',
    component: () => import('@/views/NotFoundView.vue'),
    meta: { title: 'Not found' }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior: (to, from, saved) => saved || { top: 0 }
})

router.beforeEach(async (to) => {
  const auth = useAuthStore()

  // On a hard refresh the store is empty but the backend session may still be
  // valid, so ask once before deciding where to send the user.
  if (!auth.ready) await auth.restore()

  // A server started with REQUIRE_AUTH=0 accepts every request, so putting a
  // login form in front of it locks the user out of an open deployment.
  if (to.meta.requiresAuth && auth.authRequired && !auth.isAuthenticated) {
    return { name: 'Login', query: { redirect: to.fullPath } }
  }
  if (to.meta.guestOnly && (auth.isAuthenticated || !auth.authRequired)) {
    return { name: 'Dashboard' }
  }
  return true
})

router.afterEach((to) => {
  document.title = to.meta.title
    ? `${to.meta.title} · Vision Training`
    : 'Vision Training'
})

export default router
