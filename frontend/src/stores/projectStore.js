import { defineStore } from 'pinia'
import { errorMessage, projectService } from '@/services'

export const useProjectStore = defineStore('project', {
  state: () => ({
    projects: [],
    current: null,
    images: [],
    summary: null,
    loading: false,
    error: null
  }),

  getters: {
    totalImages: (state) =>
      state.projects.reduce((sum, p) => sum + (p.total_images || 0), 0),
    totalAnnotations: (state) =>
      state.projects.reduce((sum, p) => sum + (p.total_annotations || 0), 0),
    byName: (state) => (name) => state.projects.find((p) => p.name === name) || null
  },

  actions: {
    async fetchProjects() {
      this.loading = true
      this.error = null
      try {
        this.projects = (await projectService.list()).projects
      } catch (error) {
        this.error = errorMessage(error)
      } finally {
        this.loading = false
      }
    },

    async createProject(name, description) {
      const result = await projectService.create(name, description)
      await this.fetchProjects()
      return result
    },

    async deleteProject(name) {
      const result = await projectService.remove(name)
      if (this.current?.name === name) this.current = null
      await this.fetchProjects()
      return result
    },

    async loadProject(name) {
      this.loading = true
      this.error = null
      try {
        this.current = (await projectService.get(name)).project
      } catch (error) {
        this.error = errorMessage(error)
        this.current = null
      } finally {
        this.loading = false
      }
    },

    async loadImages(name) {
      try {
        this.images = (await projectService.images(name)).images
      } catch (error) {
        this.error = errorMessage(error)
        this.images = []
      }
    },

    async loadSummary(name) {
      try {
        this.summary = await projectService.datasetSummary(name)
      } catch (error) {
        this.error = errorMessage(error)
        this.summary = null
      }
    },

    /** Reload everything a project page shows, in one round of requests. */
    async refresh(name) {
      await Promise.all([
        this.loadProject(name),
        this.loadImages(name),
        this.loadSummary(name)
      ])
    }
  }
})
